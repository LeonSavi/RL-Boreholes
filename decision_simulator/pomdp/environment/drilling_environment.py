"""
Future environment abstraction for the geological drilling POMDP.

Planned responsibilities:
- hidden geological state : the true subsurface map, unobserved by the agent
- drilling actions        : selecting and executing a borehole at a grid location
- observations            : geophysical measurements and ore readings returned by a drill
- transitions             : how the belief state evolves after each drilling action
"""
