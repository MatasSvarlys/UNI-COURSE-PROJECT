from Settings import rl_settings

step = 0

isTerminated = False
endEpisode = False
episodeCount = 0
epsilon = 0.9995
rewardsPerEpisode = []

def startNewEpisode():
    global endEpisode, episodeCount, epsilon, step, rewardsPerEpisode
    step = 0
    episodeCount += 1
    epsilon = max(epsilon - rl_settings.EPSILON_DECAY, rl_settings.MIN_EPSILON)
    rewardsPerEpisode.append(0)
    endEpisode = True
    