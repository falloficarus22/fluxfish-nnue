#ifndef TABLEBASE_H
#define TABLEBASE_H

#include <string>
#include "chess.hpp"

// Wrapper for Syzygy tablebase probing using Fathom
class Tablebase {
public:
    Tablebase();
    ~Tablebase();
    
    // Initialize tablebases from a directory
    bool init(const std::string& path);
    
    // Probe for WDL (Win/Draw/Loss) result
    // Returns: 2 = Win, 1 = Cursed Win, 0 = Draw, -1 = Blessed Loss, -2 = Loss
    // Returns TB_RESULT_FAILED if position is not in tablebase
    int probe_wdl(const chess::Board& board);
    
    // Check if position can be probed (i.e., piece count <= max pieces in TB)
    bool can_probe(const chess::Board& board) const;
    
    // Get maximum pieces in loaded tablebases
    int max_pieces() const { return max_pieces_; }
    
private:
    bool initialized_;
    int max_pieces_;
};

// Constants for tablebase results
const int TB_RESULT_FAILED = -128;
const int TB_WIN = 2;
const int TB_CURSED_WIN = 1;
const int TB_DRAW = 0;
const int TB_BLESSED_LOSS = -1;
const int TB_LOSS = -2;

#endif
