#!/bin/bash


## Evaluate the prior for testing purposes
python main.py --eval \
--total_ep 1000 --epsilon_eval 0.2 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir priors/OneDiffusion/eval

# No together reward
python main.py --train --plot \
--total_ep 100000 --epsilon 0.6 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir Independent/SameTargetReward \
--target_reward1 20 --target_reward2 20 --together_reward 0 --travel_reward -0.1 

python main.py --eval 100 \
--total_ep 100 --epsilon_eval 0 \
--prior_q experiments/Independent/SameTargetReward/q_val.pkl --save_dir Independent/SameTargetReward/eval

## Stay together reward grid search 
# for stay_together_reward in 0 1 2 3 4 5 6 7 8 9 10
for stay_together_reward in 11 12 13 14 15
do
python main.py --train --plot \
--total_ep 100000 --epsilon 0.6 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir Independent/together_test$((stay_together_reward)) \
--target_reward1 20 --target_reward2 10 --together_reward $((stay_together_reward)) --travel_reward -0.1
echo Finished $((stay_together_reward))
done 

# for stay_together_reward in 0 1 2 3 4 5 6 7 8 9 10
for stay_together_reward in 11 12 13 14 15
do
    python main.py --eval 100 \
    --total_ep 100 --epsilon_eval 0 \
    --prior_q experiments/Independent/together_test$((stay_together_reward))/q_val.pkl --save_dir Independent/together_test$((stay_together_reward))/eval
    echo Finished $((stay_together_reward))
done


# Agent 2 only cares about staying together, need prediction here
python main.py --train --eval --plot \
--total_ep 200000 --epsilon 0.6 --epsilon_eval 0.2 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir Independent/together_test3 \
--target_reward1 20 --target_reward2 0 --together_reward 2 --travel_reward -0.1 

for i in {1..100}
do
    python main.py --eval \
    --total_ep 100000 --epsilon_eval 0 \
    --prior_q experiments/Independent/together_test3/q_val.pkl --save_dir Independent/together_test3/eval$((i))
    echo Finished eval$((i))
done


## Agent 1 is predictive of agent 2
# Predict While Reward
python main.py --agent JointAgent_Predictor --predictor logistic --train --plot \
--total_ep 500000 --epsilon 0.3 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir SinglePredictor/PredictWhileReward \
--target_reward1 20 --target_reward2 20 --travel_reward -0.1 

python main.py --agent JointAgent_Predictor --predictor logistic --eval 500 \
--total_ep 100 --epsilon_eval 0 \
--prior_q experiments/SinglePredictor/PredictWhileReward/q_val.pkl \
--save_dir SinglePredictor/PredictWhileReward/eval


# Predict with fixed interval and higher predictor accuracy
python main.py --agent JointAgent_Predictor --predictor logistic --train --plot \
--total_ep 1000000 --epsilon 0.2 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir SinglePredictor/PredictFixInterval \
--target_reward1 20 --target_reward2 20 --travel_reward -0.1 

cp experiments/SinglePredictor/PredictFixInterval/predictor.joblib experiments/SinglePredictor/PredictFixInterval/eval/

python main.py --agent JointAgent_Predictor --predictor logistic --eval 500 \
--total_ep 100 --epsilon_eval 0 \
--prior_q experiments/SinglePredictor/PredictFixInterval/q_val.pkl \
--save_dir SinglePredictor/PredictFixInterval/eval

# Larger grid
python main.py --agent JointAgent_Predictor --predictor logistic --train --plot \
--total_ep 1000000 --epsilon 0.2 --grid_size 9 \
--prior_q experiments/priors/OneDiffusion/q_val_grid9.pkl --save_dir SinglePredictor/PredictFixInterval_Bigger \
--target_reward1 20 --target_reward2 20 --travel_reward -0.1 

mkdir experiments/SinglePredictor/PredictFixInterval_Bigger/eval
cp experiments/SinglePredictor/PredictFixInterval_Bigger/predictor.joblib experiments/SinglePredictor/PredictFixInterval_Bigger/eval/

python main.py --agent JointAgent_Predictor --predictor logistic --eval 500 \
--total_ep 100 --epsilon_eval 0 --grid_size 9 \
--prior_q experiments/SinglePredictor/PredictFixInterval_Bigger/q_val.pkl \
--save_dir SinglePredictor/PredictFixInterval_Bigger/eval

# Absolute Predictor: Agent 1 knows agent2's policy exactly
python main.py --agent JointAgent_AbsolutePredictor --train --plot \
--total_ep 1000000 --epsilon 0.2 \
--prior_q experiments/priors/OneDiffusion/q_val.pkl --save_dir AbsolutePredictor \
--target_reward1 20 --target_reward2 20 --travel_reward -0.1 

python main.py --agent JointAgent_AbsolutePredictor --eval 500 \
--total_ep 100 --epsilon_eval 0 \
--prior_q experiments/AbsolutePredictor/q_val.pkl --save_dir AbsolutePredictor/eval

# Bigger env
python main.py --agent JointAgent_AbsolutePredictor --train --plot \
--total_ep 1000000 --epsilon 0.3 \
--grid_size 9 --prior_q experiments/priors/OneDiffusion/q_val_grid9.pkl \
--save_dir AbsolutePredictor_Bigger \
--target_reward1 20 --target_reward2 20 --travel_reward -0.1 

python main.py --agent JointAgent_AbsolutePredictor --eval 500 \
--total_ep 100 --epsilon_eval 0 \
--grid_size 9 --prior_q experiments/AbsolutePredictor_Bigger/q_val.pkl \
--save_dir AbsolutePredictor_Bigger/eval
