conda create --name mcts python=3.12 -y
pip install vllm==0.7.1
cd /ssdscratch/byuan48/efficient_reasoning
pip install -e .
pip install ipywidgets