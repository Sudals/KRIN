"""Faithful ports of the published implementations.

The backbones in ``sota.py`` are adaptations written against one shared
interface; the code here instead keeps each author's model definition and adapts
only the input and output plumbing. The raw upstream files sit next to this one
so any deviation can be checked line by line.

Sources:
  IGNNK  Kaimaoge/IGNNK              basic_structure.py
  SATCN  Kaimaoge/SATCN              model.py
  GRIN   Graph-Machine-Learning-Group/grin   lib/nn/{models/grin,layers/*}.py
  SPIN   Graph-Machine-Learning-Group/spin   spin/{models,layers}/*.py
  KITS   Sam1224/KITS                lib/nn/models/kits.py
"""
