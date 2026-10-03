"""Deterministic, engine-independent benchmark workload generation."""
from .constants import CORPUS, CORPUS_SEED, PROTOCOL

def standard_corpus(target_chars=2200, trial_id=0):
    projects=['Atlas','Meridian','Harbor','Orchid','Cobalt','Lantern','Summit','Nimbus']
    services=['authentication','storage','networking','scheduler','catalog','telemetry','search','billing']
    regions=['north','south','east','west','central']
    states=['active','maintenance','degraded','ready']
    rows=[f"[BENCHMARK] protocol={PROTOCOL} corpus={CORPUS} trial={trial_id} seed={CORPUS_SEED + trial_id*104729}"]
    i=0
    while len('\n'.join(rows)) < target_chars:
        off=trial_id*13
        project=projects[(i*7+3+off)%len(projects)]; service=services[(i*5+1+off)%len(services)]
        region=regions[(i*3+2+off)%len(regions)]; state=states[(i*11+1+off)%len(states)]
        ref=(CORPUS_SEED + trial_id*104729 + i*7919) % 100000
        rows.append(f"[RECORD {i:05d}]\nProject: {project}\nService: {service}\nRegion: {region}\nStatus: {state}\nReference: {project[0]}-{ref:05d}\nDescription: The {service} service processes deterministic benchmark workload records for the {region} region.\n")
        i+=1
    return '\n'.join(rows)

def filler(target):
    if target<=0:return ''
    unit='benchmark context alpha beta gamma delta epsilon zeta eta theta. '; chars=target*4
    return (unit*((chars//len(unit))+1))[:chars]+'\n\n'

def practical_corpus(target_chars, trial_id=0):
    """Build meaningful deterministic context with retrieval markers for Practical V1."""
    checkpoints=[('ALPHA','ORCHID-7291'),('BRAVO','COBALT-4812'),('CHARLIE','LANTERN-5538'),('DELTA','HARBOR-1904')]
    projects=['Atlas','Meridian','Harbor','Orchid','Cobalt','Lantern','Summit','Nimbus']
    services=['authentication','storage','networking','scheduler','catalog','telemetry','search','billing']
    regions=['north','south','east','west','central']; states=['active','maintenance','degraded','ready']
    parts=[f"[BENCHMARK] protocol=devbits-practical-v1 corpus={CORPUS} trial={trial_id} seed={CORPUS_SEED + trial_id * 104729}\n"]
    marks=[int(target_chars*x) for x in (.20,.40,.60,.80)]; inserted=[False]*len(checkpoints); i=0
    while sum(map(len,parts)) < target_chars:
        current=sum(map(len,parts))
        for j,mark in enumerate(marks):
            if not inserted[j] and current>=mark:
                name,code=checkpoints[j]; parts.append(f"\n[CHECKPOINT]\nCheckpoint: {name}\nVerification code: {code}\n"); inserted[j]=True
        off=trial_id*19
        project=projects[(i*7+3+off)%len(projects)]; service=services[(i*5+1+off)%len(services)]
        region=regions[(i*3+2+off)%len(regions)]; state=states[(i*11+1+off)%len(states)]
        ref=(CORPUS_SEED+trial_id*104729+i*7919)%100000
        parts.append(f"[RECORD {i:06d}]\nProject: {project}\nService: {service}\nRegion: {region}\nStatus: {state}\nReference: {project[0]}-{ref:05d}\nDescription: The {service} service processes deterministic application records for the {region} region. Engineers use these records to diagnose behavior, review architecture, and plan safe implementation changes.\n")
        i+=1
    for j,inserted_flag in enumerate(inserted):
        if not inserted_flag:
            name,code=checkpoints[j]; parts.append(f"\n[CHECKPOINT]\nCheckpoint: {name}\nVerification code: {code}\n")
    return ''.join(parts)

def long_corpus(target_chars, trial_id=0):
    checkpoints=[('ALPHA','ORCHID-7291'),('BRAVO','COBALT-4812'),('CHARLIE','LANTERN-5538'),('DELTA','HARBOR-1904')]
    projects=['Atlas','Meridian','Harbor','Orchid','Cobalt','Lantern','Summit','Nimbus']
    services=['authentication','storage','networking','scheduler','catalog','telemetry','search','billing']
    regions=['north','south','east','west','central']; states=['active','maintenance','degraded','ready']
    parts=[f"[BENCHMARK] protocol={PROTOCOL} corpus={CORPUS} trial={trial_id} seed={CORPUS_SEED+trial_id*104729}\n"]
    marks=[int(target_chars*x) for x in (.25,.50,.75,.90)]; inserted=[False]*4; i=0
    while sum(map(len,parts)) < target_chars:
        current=sum(map(len,parts))
        for j,m in enumerate(marks):
            if not inserted[j] and current>=m:
                name,code=checkpoints[j]; parts.append(f"\n[CHECKPOINT]\nCheckpoint: {name}\nVerification code: {code}\n"); inserted[j]=True
        off=trial_id*17
        p=projects[(i*7+3+off)%len(projects)]; sv=services[(i*5+1+off)%len(services)]
        rg=regions[(i*3+2+off)%len(regions)]; st=states[(i*11+1+off)%len(states)]
        ref=(CORPUS_SEED+trial_id*104729+i*7919)%100000
        parts.append(f"[RECORD {i:06d}]\nProject: {p}\nService: {sv}\nRegion: {rg}\nStatus: {st}\nReference: {p[0]}-{ref:05d}\nDescription: The {sv} service processes deterministic workload records for the {rg} region while preserving stable benchmark structure.\n")
        i+=1
    for j in range(4):
        if not inserted[j]:
            name,code=checkpoints[j]; parts.append(f"\n[CHECKPOINT]\nCheckpoint: {name}\nVerification code: {code}\n")
    return ''.join(parts)
