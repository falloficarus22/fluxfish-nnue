#include "tablebase.h"
#include "Fathom/src/tbprobe.h"
#include <iostream>

Tablebase::Tablebase() : initialized_(false), max_pieces_(0) {}

Tablebase::~Tablebase() {
    if (initialized_) {
        tb_free();
    }
}

bool Tablebase::init(const std::string& path) {
    if (initialized_) {
        tb_free();
    }
    
    initialized_ = tb_init(path.c_str());
    
    if (initialized_) {
        max_pieces_ = TB_LARGEST;
        if (max_pieces_ > 0) {
            std::cout << "info string Syzygy tablebases initialized: " << max_pieces_ << "-piece" << std::endl;
        } else {
            std::cout << "info string Tablebase path set, but no files found" << std::endl;
        }
    } else {
        std::cout << "info string Failed to initialize tablebases from: " << path << std::endl;
    }
    
    return initialized_ && max_pieces_ > 0;
}

bool Tablebase::can_probe(const chess::Board& board) const {
    if (!initialized_ || max_pieces_ == 0) return false;
    
    // Count pieces
    int piece_count = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (board.at<chess::Piece>(chess::Square(sq)) != chess::Piece::NONE) {
            piece_count++;
        }
    }
    
    return piece_count <= max_pieces_;
}

int Tablebase::probe_wdl(const chess::Board& board) {
    if (!can_probe(board)) {
        return TB_RESULT_FAILED;
    }
    
    // Extract bitboards from our chess library
    uint64_t white = board.us(chess::Color::WHITE).getBits();
    uint64_t black = board.us(chess::Color::BLACK).getBits();
    uint64_t kings = board.pieces(chess::PieceType::KING).getBits();
    uint64_t queens = board.pieces(chess::PieceType::QUEEN).getBits();
    uint64_t rooks = board.pieces(chess::PieceType::ROOK).getBits();
    uint64_t bishops = board.pieces(chess::PieceType::BISHOP).getBits();
    uint64_t knights = board.pieces(chess::PieceType::KNIGHT).getBits();
    uint64_t pawns = board.pieces(chess::PieceType::PAWN).getBits();
    
    // Get en passant square
    unsigned ep = 0;
    if (board.enpassantSq() != chess::Square::NO_SQ) {
        ep = (unsigned)board.enpassantSq().index();
    }
    
    // Get turn
    bool turn = (board.sideToMove() == chess::Color::WHITE);
    
    // Get 50-move rule counter (half-move clock)
    unsigned rule50 = board.halfMoveClock();
    
    // Get castling rights (Fathom expects 0 if no castling possible for positions in TB)
    // Since most endgame positions don't have castling, we set to 0
    unsigned castling = 0;
    // Note: tb_probe_wdl will return TB_RESULT_FAILED if castling != 0
    
    unsigned result = tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns,
                                   rule50, castling, ep, turn);
    
    if (result == TB_RESULT_FAILED) {
        return TB_RESULT_FAILED;
    }
    
    // Convert Fathom result to our format
    unsigned wdl = TB_GET_WDL(result);
    
    switch (wdl) {
        case TB_LOSS:           return TB_LOSS;        // -2
        case TB_BLESSED_LOSS:   return TB_BLESSED_LOSS; // -1
        case TB_DRAW:           return TB_DRAW;         // 0
        case TB_CURSED_WIN:     return TB_CURSED_WIN;   // 1
        case TB_WIN:            return TB_WIN;          // 2
        default:                return TB_RESULT_FAILED;
    }
}
