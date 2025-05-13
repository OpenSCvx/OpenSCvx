from typing import List

from openscvx.constraints.ctcs import CTCSConstraint

def sort_ctcs_constraints(constraints_ctcs: List[CTCSConstraint], N: int):        
    idx_to_nodes: dict[int, tuple] = {}
    next_idx = 0
    for c in constraints_ctcs:
        # normalize None to full horizon
        c.nodes = c.nodes or (0, N)
        key = c.nodes

        if c.idx is not None:
            # user supplied an identifier: ensure it always points to the same interval
            if c.idx in idx_to_nodes:
                if idx_to_nodes[c.idx] != key:
                    raise ValueError(
                        f"idx={c.idx} was first used with interval={idx_to_nodes[c.idx]}, "
                        f"but now you gave it interval={key}"
                    )
            else:
                idx_to_nodes[c.idx] = key

        else:
            # no identifier: see if this interval already has one
            for existing_id, nodes in idx_to_nodes.items():
                if nodes == key:
                    c.idx = existing_id
                    break
            else:
                # brand-new interval: pick the next free auto-id
                while next_idx in idx_to_nodes:
                    next_idx += 1
                c.idx = next_idx
                idx_to_nodes[next_idx] = key
                next_idx += 1
    
    # Extract your intervals in ascending‐idx order
    ordered_ids       = sorted(idx_to_nodes.keys())
    node_intervals    = [ idx_to_nodes[i] for i in ordered_ids ]
    id_to_position    = { ident: pos for pos, ident in enumerate(ordered_ids) }
    num_augmented_states = len(ordered_ids)

    return constraints_ctcs, node_intervals, num_augmented_states,