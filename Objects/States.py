from Settings import rl_settings

episodeFrame = 0

isTerminated = False
endEpisode = False
episodeCount = 0
epsilon = 1
rewardsPerEpisode = {"player_one": [], "player_two": []}

def startNewEpisode():
    global endEpisode, episodeCount, epsilon, episodeFrame, rewardsPerEpisode
    episodeFrame = 0
    episodeCount += 1
    # epsilon = max(epsilon - rl_settings.EPSILON_DECAY, rl_settings.MIN_EPSILON)
    rewardsPerEpisode["player_one"].append(rl_settings.START_REWARD)
    rewardsPerEpisode["player_two"].append(rl_settings.START_REWARD)
    endEpisode = True
    with open("q_values_log.txt", "a") as f:
        f.write(f"Episode {episodeCount - 1} ended with reward: {rewardsPerEpisode['player_one'][-2] if len(rewardsPerEpisode['player_one']) > 1 else 'N/A'}\n")
    
    