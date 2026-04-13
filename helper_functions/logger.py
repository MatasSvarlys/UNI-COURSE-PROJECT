import time
from logging.handlers import QueueListener

import logging


def logging_worker(agent_queue, action_log_file):

    # Create the rewards/actions filename and the loss filename
    loss_log_file = action_log_file.replace("_log.csv", "_loss.csv")

    # Handler for Rewards/Actions (checking random actions)
    # Columns: Episode, Frame, Epsilon, IsRandom, Action, Reward
    action_handler = logging.FileHandler(action_log_file, mode='a')
    
    # Handler for Loss Values (for graphing)
    # Columns: Episode, Step, Loss
    loss_handler = logging.FileHandler(loss_log_file, mode='a')

    def dispatch_record(record):
        if ".loss" in record.name:
            loss_handler.handle(record)
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