.PHONY: setup create-dataset run-hybrid-dqn run-all clean

setup:
	mkdir -p utils logs data models

create-dataset: setup
	python create_dataset.py

run-hybrid-dqn: setup
	python hybrid_dqn_agent.py

run-all: create-dataset run-hybrid-dqn

clean:
	rm -f dataset.csv
	rm -f models/*.pth
	rm -f logs/*.log

run:
	@env/bin/python DQN_RL_agent.py

run-test:
	@env/bin/python dqn_runner.py

run-cnn-test:
	@env/bin/python dqn_cnn.py

run-cnn-models:
	@env/bin/python cnn_models.py