# AlphaFy

  A Gomoku AI based on AlphaGo Zero's Algorithm.

## Usage

  1. Create a player.
  
      ```
        from player import Player
        # This will give you a untrained random model.
        # For a pretrained model, use:
        # fy = Player('best')
        fy = Player()
      ```
    
  2. Collect data via self-playing.
  
     ```
      # This option turn on a rule-based AI as a guide, but it's very slow.
      # Turn it off when the AI seems to understand its goal.
      fy.enable_guide()
      fy.set_thinking_depth(256) # How many nodes will be expanded in MCTS.
      for i in range(5):
        fy.self_play(show_board=True)
      ```
    
  3. Save and load.
    
      ```
      # Save model and data separately.
      # It's equal to:
      # fy.save('example', override=True)
      fy.save_data('example', override=True)
      fy.save_model('example', override=True)

      fy.load_model('example')
      fy.load_data('example', merge=False) # Wheather or not overrided current data.
      ```
  4. Train the model.
  
      ```
      fy.train(epochs=5, batch_size=128)
      ```
  5. Evaluate performance.
  
      ```
      fy.vs_user(first_hand=True)

      # Or you can compare between two models.
    
      rival = Player('best')
      fy.vs(rival)
      ```
      ```
