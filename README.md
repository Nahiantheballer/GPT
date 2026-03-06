# GPT Chess Engine (Python, UCI)

A pure-Python chess engine with strong classical search features:

- Legal move generation for all standard rules
- Iterative deepening negamax alpha-beta
- Quiescence search
- Transposition table (Zobrist hashing)
- Killer + history heuristics
- UCI protocol support

## Quick start

```bash
python chess_engine.py --depth 6 --time 3
```

Run perft:

```bash
python chess_engine.py --perft 3 --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Run as UCI engine:

```bash
python chess_engine.py --uci
```

Then connect from a GUI (Arena, CuteChess, etc.).

## Testing

```bash
pytest -q
```

## Notes

This is designed to be a strong educational baseline engine in Python. For peak Elo, porting the same architecture to C++/Rust and adding NNUE is recommended.
