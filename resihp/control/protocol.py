"""Control-plane message vocabulary.

Paths: worker <-> NodeAgent (localhost) <-> Coordinator. Every message that is
part of a reconfiguration carries ``gen``; receivers drop anything older than
the generation they have already moved to (fencing of stale actors).
"""

# worker -> agent
REGISTER = "REGISTER"
HEARTBEAT = "HEARTBEAT"
ABORTED = "ABORTED"          # gen, committed_iteration, has_reduced_grads, holdings
READY = "READY"              # gen
REPORT = "REPORT"            # kind (nccl_error | failslow | health), detail

# agent -> coordinator
AGENT_REGISTER = "AGENT_REGISTER"
WORKER_REGISTER = "WORKER_REGISTER"
NODE_HEARTBEAT = "NODE_HEARTBEAT"
WORKER_FAILED = "WORKER_FAILED"
WORKER_MSG = "WORKER_MSG"

# coordinator -> agent
KILL = "KILL"
TO_WORKER = "TO_WORKER"
COORDINATOR_HELLO = "COORDINATOR_HELLO"

# coordinator -> worker (relayed by agent)
RECONFIG_NOTICE = "RECONFIG_NOTICE"   # gen, failed, immediate (abort now) or at the next iteration boundary
PREPARE = "PREPARE"                   # gen, world, payload, store_addr, store_prefix, recovery
COMMIT = "COMMIT"                     # gen
SELF_FENCE = "SELF_FENCE"             # agent lost the coordinator for longer than the lease
SHUTDOWN = "SHUTDOWN"                 # every member of the world reported done; standby workers exit too


def msg(msg_type: str, /, **payload) -> dict:
    payload["type"] = msg_type
    return payload
