"""Local resource guard for this DSS batch, never for unrelated processes.

Use host process visibility. Pin this batch to at most 75% of physical cores;
reduce that allocation or temporarily stop only its own process tree when
external load/memory pressure leaves insufficient room. No jobs are killed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import os
import signal
import sys
import time

import psutil

HERE=Path(__file__).resolve().parent
GIB=1024**3


def save(path,value):
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def utc():return datetime.now(timezone.utc).isoformat()


def allocation(policy,external_percent,total_percent,mem_available_gib,disk_free_gib,memory_hold=False):
    """Use the configured CPU capacity; physical-core units are conservative."""
    shared=external_percent>=policy['external_active_threshold_percent']
    target=policy['shared_target_total_cpu_percent'] if shared else policy['solo_target_total_cpu_percent']
    hard=policy['shared_hard_total_cpu_percent'] if shared else policy['solo_hard_total_cpu_percent']
    memory_bad=(mem_available_gib<policy['memory_pause_available_gib'] or
                (memory_hold and mem_available_gib<policy['memory_resume_available_gib']))
    budget=max(0.,target-external_percent)*policy.get('cpu_capacity_for_limits',policy['logical_cpus'])/100.
    cores=min(policy['max_our_physical_cores'],int(math.floor(budget)))
    reason='normal'
    if memory_bad:reason='low_available_memory'
    elif disk_free_gib<policy['minimum_free_disk_gib']:reason='low_disk_space'
    elif cores<1:reason='external_load_leaves_no_cpu_budget'
    elif total_percent>hard:reason='total_cpu_over_limit'
    return dict(shared=shared,target_percent=target,hard_limit_percent=hard,
        cores=max(1,cores),pause=reason!='normal',reason=reason,memory_hold=memory_bad)


def same_process(pid,created,uid):
    try:
        p=psutil.Process(int(pid))
        if abs(p.create_time()-created)>.02 or p.uids().real!=uid:return None
        if p.status()==psutil.STATUS_ZOMBIE:return None
        return p
    except psutil.NoSuchProcess:return None


class Guard:
    def __init__(self,policy):
        self.policy=policy
        if os.getuid()!=policy['uid']:raise RuntimeError('UID does not match the experiment owner')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=policy['boot_id']:
            raise RuntimeError('Server rebooted; original controller PID must not be reused')
        root=self.root()
        if root is None:raise RuntimeError('Verified experiment controller is no longer running')
        if Path(root.cwd()).resolve()!=HERE:raise RuntimeError('Controller working directory mismatch')
        self.known={};self.stopped={};self.previous_cpu={};self.previous_time=time.monotonic()
        self.shutdown=False;self.memory_hold=False;self.last_event=None;self.last_metrics=0.;self.last_live=time.monotonic()

    def root(self):
        return same_process(self.policy['controller_pid'],self.policy['controller_create_time'],self.policy['uid'])

    def owned(self):
        root=self.root()
        candidates=[root,*root.children(recursive=True)] if root is not None else []
        # Already admitted children remain in scope if a failed parent exits.
        for pid,created in list(self.known.items()):
            p=same_process(pid,created,self.policy['uid'])
            if p is not None:candidates.append(p)
        result={}
        for p in candidates:
            try:
                if p.pid==os.getpid() or p.uids().real!=self.policy['uid']:continue
                result[p.pid]=p;self.known[p.pid]=p.create_time()
            except psutil.NoSuchProcess:pass
        self.known={pid:self.known[pid] for pid in result}
        return list(result.values())

    def enforce_affinity(self,processes,cores):
        selected=self.policy['allowed_physical_cpu_representatives'][:cores]
        for p in processes:
            try:
                # Linux affinity is per thread; update all MPI/helper threads,
                # not just each process's main thread. Children inherit it.
                for thread in p.threads():os.sched_setaffinity(thread.id,selected)
            except (psutil.NoSuchProcess,ProcessLookupError):pass
        return selected

    def pause(self,processes):
        for p in reversed(processes):
            try:
                if p.pid in self.stopped:continue
                # Never claim ownership of another actor's existing stop.
                if p.status() in [psutil.STATUS_STOPPED,psutil.STATUS_TRACING_STOP]:continue
                created=p.create_time();p.send_signal(signal.SIGSTOP)
                self.stopped[p.pid]=created
            except psutil.NoSuchProcess:pass
        save(HERE/'resource_guard_stopped.json',{'uid':self.policy['uid'],'processes':self.stopped,'utc':utc()})

    def resume(self):
        for pid,created in list(self.stopped.items()):
            p=same_process(pid,created,self.policy['uid'])
            if p is not None:
                try:p.send_signal(signal.SIGCONT)
                except psutil.NoSuchProcess:pass
            del self.stopped[pid]
        save(HERE/'resource_guard_stopped.json',{'uid':self.policy['uid'],'processes':{},'utc':utc()})

    def event(self,kind,**values):
        with (HERE/'resource_guard_events.jsonl').open('a') as f:
            f.write(json.dumps({'utc':utc(),'event':kind,**values})+'\n')

    def run(self):
        lock=HERE/'resource_guard.lock'
        if lock.exists():
            data=json.loads(lock.read_text())
            if same_process(data['pid'],data['created'],self.policy['uid']) is not None:
                raise RuntimeError('A resource guard is already running')
            lock.unlink()
        with lock.open('x') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:setattr(self,'shutdown',True))
        self.event('guard_started',controller_pid=self.policy['controller_pid'],max_physical_cores=self.policy['max_our_physical_cores'])
        psutil.cpu_percent(interval=None)
        processes=self.owned()
        self.enforce_affinity(processes,self.policy['max_our_physical_cores'])
        for p in processes:
            try:self.previous_cpu[(p.pid,p.create_time())]=sum(p.cpu_times()[:2])
            except psutil.NoSuchProcess:pass
        try:
            while not self.shutdown and not (HERE/'resource_guard.stop').exists():
                time.sleep(self.policy['sample_seconds'])
                now=time.monotonic();elapsed=now-self.previous_time;self.previous_time=now
                logical_total=float(psutil.cpu_percent(interval=None));processes=self.owned()
                capacity=self.policy.get('cpu_capacity_for_limits',self.policy['logical_cpus'])
                total=logical_total*self.policy['logical_cpus']/capacity
                own_seconds=0.;current={}
                for p in processes:
                    try:
                        key=(p.pid,p.create_time());value=sum(p.cpu_times()[:2]);current[key]=value
                        # New children cannot have consumed more CPU wall time
                        # than their age; attribute their first partial sample.
                        before=self.previous_cpu.get(key,0.)
                        own_seconds+=max(0.,value-before)
                    except psutil.NoSuchProcess:pass
                self.previous_cpu=current
                ours=min(total,max(0.,100*own_seconds/elapsed/capacity))
                external=max(0.,total-ours)
                mem=psutil.virtual_memory();disk=psutil.disk_usage(str(HERE))
                decision=allocation(self.policy,external,total,mem.available/GIB,disk.free/GIB,self.memory_hold)
                self.memory_hold=decision['memory_hold']
                cpus=self.enforce_affinity(processes,decision['cores'])
                if decision['pause']:self.pause(processes)
                else:self.resume()
                signature=(decision['shared'],decision['cores'],decision['pause'],decision['reason'])
                record=dict(utc=utc(),guard_pid=os.getpid(),controller_pid=self.policy['controller_pid'],
                    controller_alive=self.root() is not None,total_cpu_percent=round(total,2),
                    cpu_limit_basis='physical-core CPU time equivalents' if capacity==self.policy['physical_cores'] else 'logical CPUs',
                    logical_cpu_percent=round(logical_total,2),
                    experiment_cpu_percent=round(ours,2),external_cpu_percent=round(external,2),
                    available_memory_gib=round(mem.available/GIB,2),memory_used_percent=mem.percent,
                    swap_used_gib=round(psutil.swap_memory().used/GIB,2),disk_free_gib=round(disk.free/GIB,1),
                    load_average=list(os.getloadavg()),controlled_process_count=len(processes),
                    paused_process_count=len(self.stopped),selected_cpu_ids=cpus,
                    physical_core_ceiling=self.policy['max_our_physical_cores'],**decision)
                save(HERE/'resource_guard_status.json',record)
                if signature!=self.last_event:
                    self.event('allocation_changed',**record);self.last_event=signature
                if now-self.last_metrics>=30:
                    with (HERE/'resource_guard_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    self.last_metrics=now
                if processes:self.last_live=now
                elif now-self.last_live>10:
                    final=HERE/'controller_status.json'
                    state=json.loads(final.read_text()) if final.exists() else {}
                    self.event('batch_stopped',controller_status=state)
                    save(HERE/'resource_guard_status.json',dict(record,stage='finished',controller_status=state))
                    break
        except Exception as exc:
            self.event('guard_error',reason=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            # Release only stops issued by this guard. Keep the conservative
            # affinity ceiling in place even when the monitor is disabled.
            self.resume();lock.unlink(missing_ok=True)
            self.event('guard_exited',affinity_ceiling_retained=True)


def self_test():
    p=dict(max_our_physical_cores=15,logical_cpus=40,cpu_capacity_for_limits=20,
        solo_target_total_cpu_percent=75,solo_hard_total_cpu_percent=80,
        shared_target_total_cpu_percent=88,shared_hard_total_cpu_percent=90,
        external_active_threshold_percent=2,memory_pause_available_gib=8,
        memory_resume_available_gib=12,minimum_free_disk_gib=20)
    a=allocation(p,0,50,80,100)
    assert a['cores']==15 and not a['pause'] and not a['shared']
    a=allocation(p,60,85,80,100)
    assert a['cores']==5 and not a['pause'] and a['shared']
    assert allocation(p,90,95,80,100)['pause']
    assert allocation(p,1,81,80,100)['reason']=='total_cpu_over_limit'
    assert allocation(p,0,10,7,100)['reason']=='low_available_memory'
    assert allocation(p,0,10,10,100,True)['pause']
    assert not allocation(p,0,10,13,100,True)['pause']
    assert allocation(p,0,10,80,10)['reason']=='low_disk_space'
    assert same_process(os.getpid(),psutil.Process().create_time()+1000,os.getuid()) is None
    assert same_process(os.getpid(),psutil.Process().create_time(),os.getuid()) is not None
    print('PASS: idle/shared CPU budgets, hard thresholds, memory hysteresis, disk pressure and PID reuse protection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    Guard(json.loads((HERE/'resource_policy.json').read_text())).run()


if __name__=='__main__':main()
