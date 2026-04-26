import time
from logging.handlers import QueueListener

import logging

from numpy import record


def logging_worker(agent_queue, action_log_file):

    # Create the rewards/actions filename and the loss filename
    loss_log_file = action_log_file.replace("_log.csv", "_loss.csv")
    qval_log_file = action_log_file.replace("_log.csv", "_qvals.csv")
    dist_log_file = action_log_file.replace("_log.csv", "_dist.csv")
    # Columns: Episode, Frame, Epsilon, IsRandom, Action, Reward
    action_handler = logging.FileHandler(action_log_file, mode='a')
    
    # Columns: Loss
    loss_handler = logging.FileHandler(loss_log_file, mode='a')

    # Columns: Q-values for each action
    qval_handler = logging.FileHandler(qval_log_file, mode='a')

    # Columns: Distribution for first action (as an example)
    dist_handler = logging.FileHandler(dist_log_file, mode='a')

    loss_buffers = {}
    
    def dispatch_record(record):
        if ".loss" in record.name:
            
            agent_id = record.name 
            if agent_id not in loss_buffers:
                loss_buffers[agent_id] = []
            
            loss_buffers[agent_id].append(float(record.msg))
            
            if len(loss_buffers[agent_id]) >= 10:
                avg = sum(loss_buffers[agent_id]) / 10
                
                # Use our styling function
                record.msg = format_loss_log(avg)
                loss_handler.handle(record)
                loss_buffers[agent_id] = []
            
        elif ".qval" in record.name:
            qval_handler.handle(record)
        elif ".dist" in record.name:
            dist_handler.handle(record)
        else:
            action_handler.handle(record)

    listener = QueueListener(agent_queue)
    listener.handle = dispatch_record
    listener.start()
    
    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

def log_action(logger, agentName, episode, frame, epsilon, isRandom, action_name, reward):
    log_msg = (f"agent: {agentName}, episode: {episode}, frame: {frame}, "
               f"epsilon: {epsilon:.10f}, random: {isRandom}, "
               f"action: {action_name}, reward: {reward:.2f}")
    logger.info(log_msg)

def log_episode_end(logger, agentName, episode, total_reward):
    log_msg = f"agent: {agentName}, SUMMARY for ep {episode}: total reward = {total_reward:.2f}"
    logger.info(log_msg)

def log_q_values(logger, episode, frame, q_array):
    # Converts the tensor/array to a comma-separated string
    q_str = ", ".join([f"{val:.6f}" for val in q_array])
    log_msg = f"episode: {episode}, frame: {frame}, q-values: {q_str}"
    logger.info(log_msg)

def format_loss_log(avg_loss):
    return f"{avg_loss:.10f}"

def log_distribution(logger, episode, frame, dist_array):
    # Just logging the first action's distribution as a sample
    dist_str = ", ".join([f"{val:.4f}" for val in dist_array[5]]) 
    log_msg = f"episode: {episode}, frame: {frame}, dist: {dist_str}"
    logger.info(log_msg)