import time
from logging.handlers import QueueListener

import logging


def logging_worker(agent_queue, log_file):
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    listener = QueueListener(agent_queue, file_handler)
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