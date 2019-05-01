from gym.envs.registration import register

register(
	id='piranhas-v0',
	entry_point='gym_piranhas.envs:PiranhasEnv',
)
