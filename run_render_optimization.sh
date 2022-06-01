mkdir -p ./optimization_gifs
rm -rf optimization_gifs/*

nohup python3 render_optimization_policies.py &> render_optimization_log.out & 
