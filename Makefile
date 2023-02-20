train-rl-singleact:
	@echo 'train a controller to perform the first control task'
	cd learn_controller/single_action && \
	python ./agent.py

eval-rl-singleact:
	@echo 'evaluate a trained controller for the task with a single action'
	cd learn_controller/single_action && python ./test_model.py

train-rl-multiact:
	@echo 'train a controller to perform the second control task'
	cd learn_controller && python agent.py

eval-rl-multiact:
	@echo 'evaluate a trained controller for the task with multiple actions'
	cd learn_controller && python test_model.py