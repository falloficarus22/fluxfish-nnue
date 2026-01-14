# FluxFish Reinforcement Learning Training

CPU-optimized reinforcement learning training system for FluxFish chess engine.

## Quick Start

### 1. Install Dependencies
```bash
pip install torch chess numpy
```

### 2. Start Training
```bash
python train_rl.py
```

Choose training mode:
- **Quick**: 10 iterations (test run)
- **Standard**: 100 iterations 
- **Full**: 1000 iterations
- **Custom**: Set your own parameters

### 3. Monitor Training
Training logs saved to `rl_training.log`
Model checkpoints saved to `fluxfish_rl.nnue`
Experience buffer saved to `rl_experience.json`

### 4. Test Trained Model
```bash
python evaluate_rl.py
```

## Training Features

### CPU Optimizations
- Smaller batch sizes (128) for memory efficiency
- Reduced MCTS iterations (200-500) for speed
- Multi-core utilization
- Experience replay for sample efficiency

### RL Components
- **Self-Play**: Engine plays against itself
- **Experience Replay**: Stores 50,000 positions
- **Temporal Difference Learning**: Learns from game outcomes
- **Exploration**: Epsilon-greedy move selection
- **Adaptive Learning**: Learning rate scheduling

### Training Process
1. **Self-Play Phase**: Play games, collect (position, value, policy) data
2. **Training Phase**: Train NNUE on collected experience
3. **Evaluation**: Monitor loss and exploration rate
4. **Checkpoint**: Save progress every 5 iterations

## Configuration

Edit `RLConfig` in `rl_train.py` to customize:
- `batch_size`: Training batch size (default: 128)
- `games_per_iteration`: Self-play games per iteration (default: 20)
- `mcts_iterations`: MCTS search depth (default: 500)
- `replay_buffer_size`: Experience buffer size (default: 50,000)
- `learning_rate`: Initial learning rate (default: 0.001)

## File Structure

```
fluxfish-nnue/
├── rl_train.py          # Main RL training pipeline
├── train_rl.py          # Training launcher
├── evaluate_rl.py       # Model evaluation
├── fluxfish_rl.nnue     # Trained model checkpoint
├── rl_experience.json   # Experience replay buffer
├── rl_training.log      # Training logs
└── nnue_model.py        # NNUE model architecture
```

## Expected Training Time

**CPU Only** (no GPU):
- Quick Training: ~30 minutes
- Standard Training: ~5 hours
- Full Training: ~50 hours

## Tips for Better Training

1. **Start Small**: Use Quick Training first to verify setup
2. **Monitor Loss**: Stop if loss plateaus below 0.001
3. **Save Often**: Checkpoints saved automatically
4. **Evaluate**: Test model quality after training
5. **Patience**: RL training takes time to show improvement

## Integration with Existing Code

- Uses existing `FluxFishNNUE` model architecture
- Compatible with current MCTS implementation
- Preserves original training files (`train_fast.py`)
- Can be deployed to Lichess using same UCI interface

## Troubleshooting

**Out of Memory**: Reduce `batch_size` or `games_per_iteration`
**Slow Training**: Reduce `mcts_iterations` or use fewer CPU cores
**Poor Moves**: Increase training iterations or adjust learning rate
**Crashes**: Check `rl_training.log` for error details
