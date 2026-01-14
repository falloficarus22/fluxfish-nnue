#include "chess_lite.h"
#include <sstream>
#include <vector>

void Position::reset() {
    for (int i = 0; i < 7; ++i) {
        pieces[i][0] = 0;
        pieces[i][1] = 0;
    }
    sideToMove = WHITE;
    castlingRights = 0;
    enPassant = -1;
    halfmoveClock = 0;
    fullmoveNumber = 1;
}

// Very basic FEN parser (limited for now)
void Position::setFen(const std::string& fen) {
    reset();
    std::stringstream ss(fen);
    std::string rankStr, stm, castling, ep;
    ss >> rankStr >> stm >> castling >> ep;

    int r = 7, f = 0;
    for (char c : rankStr) {
        if (c == '/') { r--; f = 0; }
        else if (isdigit(c)) { f += (c - '0'); }
        else {
            int sq = r * 8 + f;
            Color col = isupper(c) ? WHITE : BLACK;
            char pc = tolower(c);
            PieceType pt;
            if (pc == 'p') pt = PAWN;
            else if (pc == 'n') pt = KNIGHT;
            else if (pc == 'b') pt = BISHOP;
            else if (pc == 'r') pt = ROOK;
            else if (pc == 'q') pt = QUEEN;
            else pt = KING;
            pieces[pt][col] |= (1ULL << sq);
            f++;
        }
    }
    sideToMove = (stm == "w") ? WHITE : BLACK;
}

// Convert board to NNUE features
std::vector<float> Position::getFeatures(Color perspective) const {
    std::vector<float> features(768, 0.0f);
    for (int pt = 0; pt < 6; ++pt) {
        for (int col = 0; col < 2; ++col) {
            Bitboard bb = pieces[pt][col];
            while (bb) {
                int sq = __builtin_ctzll(bb);
                bb &= (bb - 1);

                // Perspective transformation
                int finalSq = sq;
                if (perspective == BLACK) {
                    // Mirror square vertically (sq ^ 56)
                    finalSq = sq ^ 56;
                }

                int rel_color = (col == perspective) ? 0 : 1;
                int idx = pt * 128 + rel_color * 64 + finalSq;
                if (idx >= 0 && idx < 768) features[idx] = 1.0f;
            }
        }
    }
    return features;
}

// Simple movegen (pseudo-legal + king safety check for now)
// Note: This is a placeholder for a full movegen. 
// For now, I'll use a very simple legal move set for demonstration.
std::vector<Move> Position::generateMoves() const {
    std::vector<Move> moves;
    // In a real implementation, we'd have bitboard magic here.
    // For this migration, I'll assume we integrate a movegen library 
    // or I'll provide a basic one in the next steps.
    return moves;
}

void Position::makeMove(Move m) {
    // Basic bitboard update
    // Determine piece type being moved
    PieceType pt = NONE;
    Color us = sideToMove;
    for (int i = 0; i < 6; ++i) {
        if (pieces[i][us] & (1ULL << m.from)) {
            pt = (PieceType)i;
            break;
        }
    }

    if (pt == NONE) return;

    // Remove from source, add to dest
    pieces[pt][us] &= ~(1ULL << m.from);
    pieces[pt][us] |= (1ULL << m.to);

    // Capture logic
    Color them = (us == WHITE) ? BLACK : WHITE;
    for (int i = 0; i < 6; ++i) {
        pieces[i][them] &= ~(1ULL << m.to);
    }

    // Pawn promotion
    if (m.promo != NONE) {
        pieces[PAWN][us] &= ~(1ULL << m.to);
        pieces[m.promo][us] |= (1ULL << m.to);
    }

    sideToMove = them;
}

bool Position::isGameOver() const {
    return false; 
}

float Position::getResult() const {
    return 0.5f; 
}
