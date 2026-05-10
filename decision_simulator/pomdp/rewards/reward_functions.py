"""
Future reward formulations for geological drilling decisions.

Planned reward signals:
- ore reward          : direct peak ore yield at drilled location
- information gain    : reduction in belief entropy after observing a borehole
- economic utility    : discounted value accounting for drilling cost vs. ore yield
- mine/abandon value  : terminal reward based on the final mine-or-abandon decision
"""
