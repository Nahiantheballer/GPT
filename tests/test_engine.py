from chess_engine import Engine, Position, perft, parse_uci_move


def test_perft_start_position_depth_1_to_3():
    pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert perft(pos, 1) == 20
    assert perft(pos, 2) == 400
    assert perft(pos, 3) == 8902


def test_engine_returns_legal_move():
    pos = Position("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3")
    legal = {m.uci() for m in pos.generate_legal()}
    engine = Engine()
    best, score, nodes = engine.iterative_deepening(pos, max_depth=3, max_time=0.0)
    assert best is not None
    assert best.uci() in legal
    assert isinstance(score, int)
    assert nodes > 0


def test_parse_uci_move_matches_legal_move():
    pos = Position()
    mv = parse_uci_move(pos, "e2e4")
    assert mv is not None
    assert mv.uci() == "e2e4"
