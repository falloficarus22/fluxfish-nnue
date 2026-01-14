#ifndef CHESS_LITE_H
#define CHESS_LITE_H

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

typedef uint64_t Bitboard;

enum PieceType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NONE };
enum Color { WHITE, BLACK, BOTH };

struct Move {
    int from, to;
    PieceType promo = NONE;
    
    bool operator==(const Move& other) const { return from == other.from && to == other.to && promo == other.promo; }
    std::string uci() const {
        static const char* sqs[] = {
            "a1","b1","c1","d1","e1","f1","g1","h1",
            "a2","b2","c2","d2","e2","f2","g2","h2",
            "a3","b3","c3","d3","e3","f3","g3","h3",
            "a4","b4","c4","d4","e4","f4","g4","h4",
            "a5","b5","c5","d5","e5","f5","g5","h5",
            "a6","b6","c6","d6","e6","f6","g6","h6",
            "a7","b7","c7","d7","e7","f7","g7","h7",
            "a8","b8","c8","d8","e8","f8","g8","h8"
        };
        std::string s = std::string(sqs[from]) + std::string(sqs[to]);
        if (promo == KNIGHT) s += 'n';
        if (promo == BISHOP) s += 'b';
        if (promo == ROOK) s += 'r';
        if (promo == QUEEN) s += 'q';
        return s;
    }
};

class Position {
public:
    Bitboard pieces[7][2]; // [PieceType][Color]
    Color sideToMove;
    int castlingRights;
    int enPassant;
    int halfmoveClock;
    int fullmoveNumber;

    Position() { reset(); }
    void reset();
    void setFen(const std::string& fen);
    
    std::vector<Move> generateMoves() const;
    void makeMove(Move m);
    
    // For NNUE
    std::vector<float> getFeatures(Color perspective) const;
    
    bool isGameOver() const;
    float getResult() const; // 1.0, 0.5, 0.0 relative to sideToMove
};

#endif
