

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedDeque;


/**
 * Component of the Chess game that detects check mates in the game.
 *
 */
public class CheckmateDetector {
    private Board b;
    private LinkedList<Square> movableSquares;
    private final LinkedList<Square> squares;
    public HashMap<Integer,List<Piece>> wMoves;
    public HashMap<Integer,List<Piece>> bMoves;
    
    /**
     * Constructs a new instance of CheckmateDetector on a given board. By
     * convention should be called when the board is in its initial state.
     * 
     * @param b The board which the detector monitors
     */
    public CheckmateDetector(Board b /*, LinkedList<Piece> wPieces,
            LinkedList<Piece> bPieces, King wk, King bk*/) {
        this.b = b;

        // Initialize other fields
        squares = new LinkedList<Square>();
        movableSquares = new LinkedList<Square>();
        wMoves = new HashMap<Integer,List<Piece>>();
        bMoves = new HashMap<Integer,List<Piece>>();
        
        Square[][] brd = b.getSquareArray();
        
        // add all squares to squares list and as hashmap keys
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                squares.add(brd[y][x]);
                wMoves.put(brd[y][x].hashCode(), new LinkedList<Piece>());
                bMoves.put(brd[y][x].hashCode(), new LinkedList<Piece>());
            }
        }
        
        // update situation
        update();
    }
    
    /**
     * Updates the object with the current situation of the game.
     */
    public void update() {
        // Iterators through pieces
        Iterator<Piece> wIter = b.Wpieces.iterator();
        Iterator<Piece> bIter = b.Bpieces.iterator();
        
        // empty moves and movable squares at each update
        for (List<Piece> pieces : wMoves.values()) {
            pieces.removeAll(pieces);
        }
        
        for (List<Piece> pieces : bMoves.values()) {
            pieces.removeAll(pieces);
        }
        
        movableSquares.removeAll(movableSquares);
        
        // Add each move white and black can make to map
        while (wIter.hasNext()) {
            Piece p = wIter.next();

            if (!p.getClass().equals(King.class)) {
                if (p.getPosition() == null) {
                    wIter.remove();
                    continue;
                }

                List<Square> mvs = p.getLegalMoves(b);
                Iterator<Square> iter = mvs.iterator();
                while (iter.hasNext()) {
                    List<Piece> pieces = wMoves.get(iter.next().hashCode());
                    pieces.add(p);
                }
            }
        }
        
        while (bIter.hasNext()) {
            Piece p = bIter.next();
            
            if (!p.getClass().equals(King.class)) {
                if (p.getPosition() == null) {
                    bIter.remove();
                    continue;
                }
                
                List<Square> mvs = p.getLegalMoves(b);
                Iterator<Square> iter = mvs.iterator();
                while (iter.hasNext()) {
                    List<Piece> pieces = bMoves.get(iter.next().hashCode());
                    pieces.add(p);
                }
            }
        }
    }
    
    /**
     * Checks if the black king is threatened
     * @return boolean representing whether the black king is in check.
     */
    public boolean blackInCheck() {
        update();
        movableSquares.addAll(squares);
        Square sq = b.Bk.getPosition();
        if (wMoves.get(sq.hashCode()).isEmpty()) {
            return false;
        } else {
            return true;
        }
    }
    
    /**
     * Checks if the white king is threatened
     * @return boolean representing whether the white king is in check.
     */
    public boolean whiteInCheck() {
        update();
        movableSquares.addAll(squares);
        Square sq = b.Wk.getPosition();
        if (bMoves.get(sq.hashCode()).isEmpty()) {
            return false;
        } else {
            return true;
        }
    }
    
    /**
     * Checks whether black is in checkmate.
     * @return boolean representing if black player is checkmated.
     */
    public boolean blackCheckMated() {
        boolean checkmate = true;
        // Check if black is in check
        if (!this.blackInCheck()) return false;
        
        // If yes, check if king can evade
        if (canEvade(wMoves, b.Bk)) checkmate = false;
        
        // If no, check if threat can be captured
        List<Piece> threats = wMoves.get(b.Bk.getPosition().hashCode());
        if (canCapture(bMoves, threats, b.Bk)) checkmate = false;
        
        // If no, check if threat can be blocked
        if (canBlock(threats, bMoves, b.Bk)) checkmate = false;
        
        // If no possible ways of removing check, checkmate occurred
        return checkmate;
    }
    
    /**
     * Checks whether white is in checkmate.
     * @return boolean representing if white player is checkmated.
     */
    public boolean whiteCheckMated() {
        boolean checkmate = true;
        // Check if white is in check
        if (!this.whiteInCheck()) return false;
        
        // If yes, check if king can evade
        if (canEvade(bMoves, b.Wk)) checkmate = false;
        
        // If no, check if threat can be captured
        List<Piece> threats = bMoves.get(b.Wk.getPosition().hashCode());
        if (canCapture(wMoves, threats, b.Wk)) checkmate = false;
        
        // If no, check if threat can be blocked
        if (canBlock(threats, wMoves, b.Wk)) checkmate = false;
        
        // If no possible ways of removing check, checkmate occurred
        return checkmate;
    }
    
    /*
     * Helper method to determine if the king can evade the check.
     * Gives a false positive if the king can capture the checking piece.
     */
    private boolean canEvade(Map<Integer,List<Piece>> tMoves, King tKing) {
        boolean evade = false;
        List<Square> kingsMoves = tKing.getLegalMoves(b);
        Iterator<Square> iterator = kingsMoves.iterator();
        
        // If king is not threatened at some square, it can evade
        while (iterator.hasNext()) {
            Square sq = iterator.next();
            if (!testMove(tKing, sq)) continue;
            if (tMoves.get(sq.hashCode()).isEmpty()) {
                movableSquares.add(sq);
                evade = true;
            }
        }
        
        return evade;
    }
    
    /*
     * Helper method to determine if the threatening piece can be captured.
     */
    private boolean canCapture(Map<Integer,List<Piece>> poss,
            List<Piece> threats, King k) {
        
        boolean capture = false;
        if (threats.size() == 1) {
            Square sq = threats.get(0).getPosition();
            
            if (k.getLegalMoves(b).contains(sq)) {
                movableSquares.add(sq);
                if (testMove(k, sq)) {
                    capture = true;
                }
            }
            
            List<Piece> caps = poss.get(sq.hashCode());
            ConcurrentLinkedDeque<Piece> capturers = new ConcurrentLinkedDeque<Piece>();
            capturers.addAll(caps);
            
            if (!capturers.isEmpty()) {
                movableSquares.add(sq);
                for (Piece p : capturers) {
                    if (testMove(p, sq)) {
                        capture = true;
                    }
                }
            }
        }
        
        return capture;
    }
    
    /*
     * Helper method to determine if check can be blocked by a piece.
     */
    private boolean canBlock(List<Piece> threats, 
            Map <Integer,List<Piece>> blockMoves, King k) {
        boolean blockable = false;
        
        if (threats.size() == 1) {
            Square ts = threats.get(0).getPosition();
            Square ks = k.getPosition();
            Square[][] brdArray = b.getSquareArray();
            
            if (ks.getXNum() == ts.getXNum()) {
                int max = Math.max(ks.getYNum(), ts.getYNum());
                int min = Math.min(ks.getYNum(), ts.getYNum());
                
                for (int i = min + 1; i < max; i++) {
                    List<Piece> blks = 
                            blockMoves.get(brdArray[i][ks.getXNum()].hashCode());
                    ConcurrentLinkedDeque<Piece> blockers = 
                            new ConcurrentLinkedDeque<Piece>();
                    blockers.addAll(blks);
                    
                    if (!blockers.isEmpty()) {
                        movableSquares.add(brdArray[i][ks.getXNum()]);
                        
                        for (Piece p : blockers) {
                            if (testMove(p,brdArray[i][ks.getXNum()])) {
                                blockable = true;
                            }
                        }
                        
                    }
                }
            }
            
            if (ks.getYNum() == ts.getYNum()) {
                int max = Math.max(ks.getXNum(), ts.getXNum());
                int min = Math.min(ks.getXNum(), ts.getXNum());
                
                for (int i = min + 1; i < max; i++) {
                    List<Piece> blks = 
                            blockMoves.get(brdArray[ks.getYNum()][i].hashCode());
                    ConcurrentLinkedDeque<Piece> blockers = 
                            new ConcurrentLinkedDeque<Piece>();
                    blockers.addAll(blks);
                    
                    if (!blockers.isEmpty()) {
                        
                        movableSquares.add(brdArray[ks.getYNum()][i]);
                        
                        for (Piece p : blockers) {
                            if (testMove(p, brdArray[ks.getYNum()][i])) {
                                blockable = true;
                            }
                        }
                        
                    }
                }
            }

            if (threats.isEmpty())
                return true;

            Class<? extends Piece> tC = threats.get(0).getClass();
            
            if (tC.equals(Queen.class) || tC.equals(Bishop.class)) {
                int kX = ks.getXNum();
                int kY = ks.getYNum();
                int tX = ts.getXNum();
                int tY = ts.getYNum();
                
                if (kX > tX && kY > tY) {
                    for (int i = tX + 1; i < kX; i++) {
                        tY++;
                        List<Piece> blks = 
                                blockMoves.get(brdArray[tY][i].hashCode());
                        ConcurrentLinkedDeque<Piece> blockers = 
                                new ConcurrentLinkedDeque<Piece>();
                        blockers.addAll(blks);
                        
                        if (!blockers.isEmpty()) {
                            movableSquares.add(brdArray[tY][i]);
                            
                            for (Piece p : blockers) {
                                if (testMove(p, brdArray[tY][i])) {
                                    blockable = true;
                                }
                            }
                        }
                    }
                }
                
                if (kX > tX && tY > kY) {
                    for (int i = tX + 1; i < kX; i++) {
                        tY--;
                        List<Piece> blks = 
                                blockMoves.get(brdArray[tY][i].hashCode());
                        ConcurrentLinkedDeque<Piece> blockers = 
                                new ConcurrentLinkedDeque<Piece>();
                        blockers.addAll(blks);
                        
                        if (!blockers.isEmpty()) {
                            movableSquares.add(brdArray[tY][i]);
                            
                            for (Piece p : blockers) {
                                if (testMove(p, brdArray[tY][i])) {
                                    blockable = true;
                                }
                            }
                        }
                    }
                }
                
                if (tX > kX && kY > tY) {
                    for (int i = tX - 1; i > kX; i--) {
                        tY++;
                        List<Piece> blks = 
                                blockMoves.get(brdArray[tY][i].hashCode());
                        ConcurrentLinkedDeque<Piece> blockers = 
                                new ConcurrentLinkedDeque<Piece>();
                        blockers.addAll(blks);
                        
                        if (!blockers.isEmpty()) {
                            movableSquares.add(brdArray[tY][i]);
                            
                            for (Piece p : blockers) {
                                if (testMove(p, brdArray[tY][i])) {
                                    blockable = true;
                                }
                            }
                        }
                    }
                }
                
                if (tX > kX && tY > kY) {
                    for (int i = tX - 1; i > kX; i--) {
                        tY--;
                        List<Piece> blks = 
                                blockMoves.get(brdArray[tY][i].hashCode());
                        ConcurrentLinkedDeque<Piece> blockers = 
                                new ConcurrentLinkedDeque<Piece>();
                        blockers.addAll(blks);
                        
                        if (!blockers.isEmpty()) {
                            movableSquares.add(brdArray[tY][i]);
                            
                            for (Piece p : blockers) {
                                if (testMove(p, brdArray[tY][i])) {
                                    blockable = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return blockable;
    }
    
    /**
     * Method to get a list of allowable squares that the player can move.
     * Defaults to all squares, but limits available squares if player is in
     * check.
     * @param b boolean representing whether it's white player's turn (if yes,
     * true)
     * @return List of squares that the player can move into.
     */
    public List<Square> getAllowableSquares(boolean b) {
        movableSquares.removeAll(movableSquares);
        if (whiteInCheck()) {
            whiteCheckMated();
        } else if (blackInCheck()) {
            blackCheckMated();
        }
        return movableSquares;
    }
    
    /**
     * Tests a move a player is about to make to prevent making an illegal move
     * that puts the player in check.
     * @param p Piece moved
     * @param sq Square to which p is about to move
     * @return false if move would cause a check
     */
    public boolean testMove(Piece p, Square sq) {
        boolean movetest = true;
        // Backup current Move, so it can be undone later
        Square currSq = null;
        Piece capturedPiece = null;
        if (sq != null) {
            if (sq.isOccupied()) {
                capturedPiece = sq.getOccupyingPiece();
            }
            currSq = p.getPosition();
        }

        // Move piece to new square
        p.move(sq);
        update();

        // Check if Black or White King is in Check
        if (p.getColor() == 0 && blackInCheck()) movetest = false;
        else if (p.getColor() == 1 && whiteInCheck()) movetest = false;

        // Undo the move
        p.move(currSq);
        if (capturedPiece != null) {
            if ((capturedPiece.getColor() == 0) && (!b.Bpieces.contains(capturedPiece))) {
                b.Bpieces.add(capturedPiece);
            }
            else if ((capturedPiece.getColor() == 1) && (!b.Wpieces.contains(capturedPiece))) {
                b.Wpieces.add(capturedPiece);
            }
            capturedPiece.move(sq);
        }
        update();

        movableSquares.addAll(squares);

        return movetest;
    }

}
