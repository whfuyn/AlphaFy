# AlphaFy

  AlphaFy is a Gomoku AI based on AlphaGo Zero's Algorithm. It use the same structure as AlphaGo Zero (Monte Carlo Tree Search and Residual Network). 
  The game's board size is scalable, so you can try your algorithm on 3x3 Tic-tac-toe and then scale it to a biggger board like a 9x9 gomoku.

## Requirement
1. Python 3.6
2. Keras 

## Usage

  1. Create a player.
  
      ```python
        from player import Player
        # This will give you a untrained random model.
        # For a pretrained model, use:
        # fy = Player('latest')
        fy = Player()
      ```
    
  2. Collect data via self-playing.
  
     ```python
      # This option turn on a rule-based AI as a guide, but it's very slow.
      # Turn it off when the AI seems to understand its goal.
      fy.enable_guide()
      # This indicate how many nodes in Monte Carlo Tree will be expanded to pick each move. 
      fy.set_thinking_depth(128)
      
      # Data will be collected after each game ended.
      fy.self_play(show_board=True)
      ```
    
  3. Save and load.
    
      ```python
      # Save model and data separately.
      # It's equal to:
      # fy.save('example', override=True)
      fy.save_data('example', override=True)
      fy.save_model('example', override=True)

      fy.load_model('example')
      # If merge=False, it will override current data.
      fy.load_data('example', merge=False)
      ```
  4. Train the model.
  
      ```python
      # Note that it needs a lot of data to obtain a reasonable performance.
      fy.train(epochs=5, batch_size=128)
      ```
  5. Evaluate performance.
  
      ```python
      fy.vs_user(first_hand=True)

      # Or you can compare between two models.
    
      rival = Player('latest')
      fy.vs(rival)
      ```

## Example

No-guide, thinking-depth=512
```
     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . . . . . . . |
 6 | . . . . . . . . . |
 5 | . . . . . . . . . |
 4 | . . . . . . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . . . . . . . |
 6 | . . . . . . . . . |
 5 | . . . . . O). . . |
 4 | . . . . . . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . . . X). . . |
 6 | . . . . . . . . . |
 5 | . . . . . O . . . |
 4 | . . . . . . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . . . X . . . |
 6 | . . . . . . . . . |
 5 | . . . . . O . . . |
 4 | . . . . O). . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X). X . . . |
 6 | . . . . . . . . . |
 5 | . . . . . O . . . |
 4 | . . . . O . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O)X . . . |
 6 | . . . . . . . . . |
 5 | . . . . . O . . . |
 4 | . . . . O . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . . . . . . |
 5 | . . . . . O . . . |
 4 | X). . . O . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . . . O). . |
 5 | . . . . . O . . . |
 4 | X . . . O . . . . |
 3 | . . . . . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . . . O . . |
 5 | . . . . . O . . . |
 4 | X . . . O . . . . |
 3 | . . . X). . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . . O)O . . |
 5 | . . . . . O . . . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . . O O . . |
 5 | . . . . . O X). . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . O)O O . . |
 5 | . . . . . O X . . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . . O O O . . |
 5 | . . . . X)O X . . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . . O)O O O . . |
 5 | . . . . X O X . . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . X)O O O O . . |
 5 | . . . . X O X . . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9

     1 2 3 4 5 6 7 8 9
   +-------------------+
 9 | . . . . . . . . . |
 8 | . . . . . . . . . |
 7 | . . . X O X . . . |
 6 | . . X O O O O O). |
 5 | . . . . X O X . . |
 4 | X . . . O . . . . |
 3 | . . . X . . . . . |
 2 | . . . . . . . . . |
 1 | . . . . . . . . . |
   +-------------------+
     1 2 3 4 5 6 7 8 9
```