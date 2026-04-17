from Settings import rl_settings

episodeFrame = 0

isTerminated = False
episodeCount = 0
epsilon = 1
episodeReward = {agentName: rl_settings.START_REWARD for agentName in rl_settings.RL_CONTROL if rl_settings.RL_CONTROL[agentName]}
framesLeft = rl_settings.MAX_FRAMES


def startNewEpisode():
    global endEpisode, episodeCount, episodeFrame, episodeReward, isTerminated, framesLeft
    episodeFrame = 0
    episodeCount += 1
    isTerminated = False
    framesLeft = rl_settings.MAX_FRAMES
    for agentName in episodeReward:
        episodeReward[agentName] = rl_settings.START_REWARD