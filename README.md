

# Snake AI

A Deep Q-Learning Agent trained on the classic game of snake.

![Demo Video](images/demo.gif)

### Training
To train a model, run `python train.py`. It will output saved checkpoint models every 100 episodes (games) and a final model in the `saved` folder.

### Testing
To test a trained model with a display, run `python testmodel.py [model path]`. e.g. `python testmodel.py example_model.pth`. This opens a pygame window and continuously runs the model in a snake game until CTRL+C is pressed in the terminal.

### Todos
- Right now the snake nearly always dies because it boxes itself in, planning to feed additional data to the model during training so it can avoid this
- Also can add a small reward for moving closer to the apple and a small punishment for moving away from it to prioritize efficiency
