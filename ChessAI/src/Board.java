

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.*;
import java.util.List;
import javafx.util.Pair;
import javax.swing.*;

@SuppressWarnings("serial")
public class Board extends JPanel implements MouseListener, MouseMotionListener {
	// Resource location constants for piece images
    private static final String RESOURCES_WBISHOP_PNG = "resources/wbishop.png";
	private static final String RESOURCES_BBISHOP_PNG = "resources/bbishop.png";
	private static final String RESOURCES_WKNIGHT_PNG = "resources/wknight.png";
	private static final String RESOURCES_BKNIGHT_PNG = "resources/bknight.png";
	private static final String RESOURCES_WROOK_PNG = "resources/wrook.png";
	private static final String RESOURCES_BROOK_PNG = "resources/brook.png";
	private static final String RESOURCES_WKING_PNG = "resources/wking.png";
	private static final String RESOURCES_BKING_PNG = "resources/bking.png";
	private static final String RESOURCES_BQUEEN_PNG = "resources/bqueen.png";
	private static final String RESOURCES_WQUEEN_PNG = "resources/wqueen.png";
	private static final String RESOURCES_WPAWN_PNG = "resources/wpawn.png";
	private static final String RESOURCES_BPAWN_PNG = "resources/bpawn.png";

    private static final int DEPTH_LEVEL = 3;

	// Logical and graphical representations of board
	private Square[][] board;
    private GameWindow g;
    
    // List of pieces and whether they are movable
    public LinkedList<Piece> Bpieces;
    public LinkedList<Piece> Wpieces;
    public King Wk;
    public King Bk;

    public List<Square> movable;

    private boolean whiteTurn;

    private Piece currPiece;
    private int currX;
    private int currY;
    
    private CheckmateDetector cmd;

    public Board(GameWindow g) {
        initializeBoard(g);
    }

    public Board(GameWindow g, boolean init) {
        if (init) {
            initializeBoard(g);
        }
        else {
            this.g = g;
            board = new Square[8][8];
            whiteTurn = true;
        }
    }

    public void initializeBoard(GameWindow g) {
        this.g = g;
        board = new Square[8][8];
        Bpieces = new LinkedList<Piece>();
        Wpieces = new LinkedList<Piece>();
        setLayout(new GridLayout(8, 8, 0, 0));

        this.addMouseListener(this);
        this.addMouseMotionListener(this);

        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                int xMod = x % 2;
                int yMod = y % 2;

                if ((xMod == 0 && yMod == 0) || (xMod == 1 && yMod == 1)) {
                    board[x][y] = new Square(this, 1, y, x);
                    this.add(board[x][y]);
                } else {
                    board[x][y] = new Square(this, 0, y, x);
                    this.add(board[x][y]);
                }
            }
        }

        initializePieces();

        this.setPreferredSize(new Dimension(400, 400));
        this.setMaximumSize(new Dimension(400, 400));
        this.setMinimumSize(this.getPreferredSize());
        this.setSize(new Dimension(400, 400));

        whiteTurn = true;

    }

    private void initializePieces() {
    	
        for (int x = 0; x < 8; x++) {
            board[1][x].put(new Pawn(0, board[1][x], RESOURCES_BPAWN_PNG));
            board[6][x].put(new Pawn(1, board[6][x], RESOURCES_WPAWN_PNG));
        }
        
        board[7][3].put(new Queen(1, board[7][3], RESOURCES_WQUEEN_PNG));
        board[0][3].put(new Queen(0, board[0][3], RESOURCES_BQUEEN_PNG));
        
        Bk = new King(0, board[0][4], RESOURCES_BKING_PNG);
        Wk = new King(1, board[7][4], RESOURCES_WKING_PNG);
        board[0][4].put(Bk);
        board[7][4].put(Wk);

        board[0][0].put(new Rook(0, board[0][0], RESOURCES_BROOK_PNG));
        board[0][7].put(new Rook(0, board[0][7], RESOURCES_BROOK_PNG));
        board[7][0].put(new Rook(1, board[7][0], RESOURCES_WROOK_PNG));
        board[7][7].put(new Rook(1, board[7][7], RESOURCES_WROOK_PNG));

        board[0][1].put(new Knight(0, board[0][1], RESOURCES_BKNIGHT_PNG));
        board[0][6].put(new Knight(0, board[0][6], RESOURCES_BKNIGHT_PNG));
        board[7][1].put(new Knight(1, board[7][1], RESOURCES_WKNIGHT_PNG));
        board[7][6].put(new Knight(1, board[7][6], RESOURCES_WKNIGHT_PNG));

        board[0][2].put(new Bishop(0, board[0][2], RESOURCES_BBISHOP_PNG));
        board[0][5].put(new Bishop(0, board[0][5], RESOURCES_BBISHOP_PNG));
        board[7][2].put(new Bishop(1, board[7][2], RESOURCES_WBISHOP_PNG));
        board[7][5].put(new Bishop(1, board[7][5], RESOURCES_WBISHOP_PNG));
        
        
        for(int y = 0; y < 2; y++) {
            for (int x = 0; x < 8; x++) {
                Bpieces.add(board[y][x].getOccupyingPiece());
                Wpieces.add(board[7-y][x].getOccupyingPiece());
            }
        }
        
        cmd = new CheckmateDetector(this /*, Wpieces, Bpieces, wk, bk*/);
    }

    public Square[][] getSquareArray() {
        return this.board;
    }

    public boolean getTurn() {
        return whiteTurn;
    }

    public void setCurrPiece(Piece p) {
        this.currPiece = p;
    }

    public Piece getCurrPiece() {
        return this.currPiece;
    }

    @Override
    public void paintComponent(Graphics g) {
        // super.paintComponent(g);

        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                Square sq = board[y][x];
                if (sq != null) {
                    sq.paintComponent(g);
                }
            }
        }

        if (currPiece != null) {
            if ((currPiece.getColor() == 1 && whiteTurn)
                    || (currPiece.getColor() == 0 && !whiteTurn)) {
                final Image i = currPiece.getImage();
                if (i != null) {
                    g.drawImage(i, currX, currY, null);
                }
            }
        }
    }

    @Override
    public void mousePressed(MouseEvent e) {
        currX = e.getX();
        currY = e.getY();

        Square sq = (Square) this.getComponentAt(new Point(e.getX(), e.getY()));

        if (sq.isOccupied()) {
            currPiece = sq.getOccupyingPiece();
            if (currPiece.getColor() == 0 && whiteTurn)
                return;
            if (currPiece.getColor() == 1 && !whiteTurn)
                return;
            sq.setDisplay(false);
        }
        repaint();
    }

    private Board copyBoard() {
        Board newBoard = new Board(g, false);

        int i = 0;
        int j = 0;

        // Reposition the pieces based on Board being copied
        King wk = null;
        King bk = null;
        newBoard.Bpieces = new LinkedList<Piece>();
        newBoard.Wpieces = new LinkedList<Piece>();
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                newBoard.board[i][j] = new Square(newBoard, 0, j, i);
                Piece piece = this.board[i][j].getOccupyingPiece();
                if (piece != null) {
                    Piece newPiece = piece.copyPiece();
                    newBoard.board[i][j].put(newPiece);
                    if (piece.getColor() == 0) {
                        newBoard.Bpieces.add(newPiece);
                        if (newPiece.getClass().getName().equals("King")) {
                            bk = (King) newPiece;
                        }
                    }
                    else {
                        newBoard.Wpieces.add(newPiece);
                        if (newPiece.getClass().getName().equals("King")) {
                            wk = (King) newPiece;
                        }
                    }
                    if (piece.equals(currPiece)) {
                        newBoard.currPiece = newPiece;
                    }
                }
            }
        }

        newBoard.currX = this.currX;
        newBoard.currY = this.currY;
        newBoard.whiteTurn = this.whiteTurn;

        newBoard.cmd = new CheckmateDetector(newBoard /*, newBoard.Wpieces, newBoard.Bpieces, wk, bk*/);

        return newBoard;
    }

    private Square FindSquare(Square sq) {

        return this.board[sq.getYNum()][sq.getXNum()];
    }

    private Piece FindPiece(Piece chessPiece) {

        int i = 0;
        int j = 0;

        for (i = 0; i < this.board.length; i++) {
            for (j = 0; j < this.board[i].length; j++) {
                Piece piece = this.board[i][j].getOccupyingPiece();
                if (piece != null) {
                    if (piece.getColor() == chessPiece.getColor()) {
                        if (piece.getClass().getName().equals(chessPiece.getClass().getName())) {
                            if ((piece.getPosition().getXNum() == chessPiece.getPosition().getXNum()) &&
                                    (piece.getPosition().getYNum() == chessPiece.getPosition().getYNum())) {
                                return piece;
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

    private int MinMax_CalcVal(boolean turnSelector) {

        int i = 0;
        int j = 0;
        int valMinMax = 0;

        int pUnderThreat = 0;
        int kingUnderThreat = 0;

        int majorPieceCount = 0;
        int minorPieceCount = 0;
        boolean kingPresent = false;

        LinkedList<Piece> pieces = null;
        if (turnSelector == false) { // Black's Turn
            pieces = Bpieces;
        }
        else { // White's Turn
            pieces = Wpieces;
        }

        int pCount = pieces.size();

        for (i = 0; i < pCount; i++) {
            Piece piece = pieces.get(i);
            if (piece.getClass().getName().equals("Bishop")) { minorPieceCount++; }
            if (piece.getClass().getName().equals("Knight")) { minorPieceCount++; }
            if (piece.getClass().getName().equals("Queen")) { majorPieceCount++; }
            if (piece.getClass().getName().equals("Rook")) { majorPieceCount++; }
            if (piece.getClass().getName().equals("King")) { majorPieceCount++; kingPresent = true; }

            List<Square> possibleMoves = piece.getLegalMoves(this);
            for (j = 0; j < possibleMoves.size(); j++) {
                Square sq = possibleMoves.get(j);
                Piece occPiece = sq.getOccupyingPiece();
                if (occPiece != null)
                {
                    if ((turnSelector) && (occPiece.getColor() == 0)) { // White Piece attacking Black Piece
                        pUnderThreat++;
                        if (occPiece.getClass().getName().equals("King")){
                            kingUnderThreat++;
                        }
                    }
                    if ((!turnSelector) && (occPiece.getColor() == 1)) { // Black Piece attacking White Piece
                        pUnderThreat++;
                        if (occPiece.getClass().getName().equals("King")){
                            kingUnderThreat++;
                        }
                    }
                }
            }
        }

        valMinMax = (pCount * 4) + (majorPieceCount * 8) + (minorPieceCount * 6) +
                (pUnderThreat * 9) + (kingUnderThreat * 10) + (kingPresent ? 10 : -100);

        return valMinMax;
    }

    private Pair<Integer, Pair<Piece, Square>> MinMax_SelectPiece(boolean turnSelector, int depthLevel, String prevPos, Stack<String> futureMoves) {
        LinkedList<Piece> pieces = null;
        if (turnSelector) { pieces = Wpieces; }
        else { pieces = Bpieces; }

        Stack<String> bestFutureMoves = new Stack<String>();

        int oppMinMaxVal = 0;
        Square oppSq = null;
        Piece oppPiece = null;
        for (int i = 0; i < pieces.size(); i++) {
            Piece pieceTemp = pieces.get(i);
            if ((pieceTemp != null) && ((turnSelector && (pieceTemp.getColor() == 1)) ||
                    (!turnSelector && (pieceTemp.getColor() == 0))))
            {
                Stack<String> tempFutureMoves = new Stack<String>();
                Pair<Integer, Square> r =
                        MinMax_SelectSquare(pieceTemp, turnSelector, depthLevel + 1,
                                prevPos + pieceTemp.getPositionName() + "\r\n", tempFutureMoves);
                int valMinMax = r.getKey();
                Square sqTempNext = r.getValue();
                if ((turnSelector) && ((oppSq == null) || (valMinMax < oppMinMaxVal))) {
                    oppSq = sqTempNext;
                    oppPiece = pieceTemp;
                    oppMinMaxVal = valMinMax;
                    bestFutureMoves.removeAllElements();
                    bestFutureMoves.addAll(tempFutureMoves);
                }
                if ((!turnSelector) && ((oppSq == null) || (valMinMax > oppMinMaxVal))) {
                    oppSq = sqTempNext;
                    oppPiece = pieceTemp;
                    oppMinMaxVal = valMinMax;
                    bestFutureMoves.removeAllElements();
                    bestFutureMoves.addAll(tempFutureMoves);
                }
                if (depthLevel == 0) {
                    if (pieceTemp.getClass().getName().equals("Pawn")) {
                        ((Pawn) pieceTemp).setMoved(false); // Reset Pawn's First Move Status
                    }
                }
            }
        }

        String oppSqPosName = (oppSq == null) ? "---" : oppSq.getPositionName();
        futureMoves.addAll(bestFutureMoves);
        futureMoves.push(oppPiece.getPositionName() +
                " to " + oppSqPosName +
                " Level: " + Integer.toString(depthLevel) +
                " Val: " + Integer.toString(oppMinMaxVal));

        return (new Pair<Integer, Pair<Piece, Square>>(oppMinMaxVal, new Pair<Piece, Square>(oppPiece, oppSq)));
    }

    private Pair<Integer, Square> MinMax_SelectSquare(Piece chessPiece, boolean turnSelector, int depthLevel, String prevPos, Stack<String> futureMoves) {

        // Get the Game Tree Depth from UI
        String strDepth = g.depth.getText();
        int gameTreeDepth = Integer.parseInt(strDepth.substring(17).trim());
        gameTreeDepth = (gameTreeDepth > 0) ? gameTreeDepth : DEPTH_LEVEL;
        if (depthLevel > gameTreeDepth) {
            // Return MinMax Value if Depth Limit has reached
            int valMinMax = MinMax_CalcVal(turnSelector);
            return new Pair<Integer, Square>(valMinMax, null);
        }

        int nextMoveMinMax = 0;
        Square nextMoveSq = null;
        Board nextMoveBoard = null;
        Piece nextMoveChessPiece = null;
        Square bestNextMove = null;
        Stack<String> bestFutureMoves = new Stack<String>();

        List<Square> possibleMoves = chessPiece.getLegalMoves(this);

        // Find the next square to occupy
        for (int j = 0; j < possibleMoves.size(); j++) {
            Square sq = possibleMoves.get(j);
            // If the square is empty or
            // occupied by a white piece when its Black's Turn or
            // occupied by a black piece when its White's Turn
            if ((sq.getOccupyingPiece() == null) ||
                    ((turnSelector == false) && (sq.getOccupyingPiece().getColor() == 1)) ||
                    ((turnSelector == true) && (sq.getOccupyingPiece().getColor() == 0))
                    )
            {
                Stack<String> tempFutureMoves = new Stack<String>();

                // Backup current Move, so it can be undone later
                Square currSq = null;
                Piece capturedPiece = null;
                if (sq != null) {
                    if (sq.isOccupied()) {
                        capturedPiece = sq.getOccupyingPiece();
                    }
                    currSq = chessPiece.getPosition();
                }

                boolean success = this.takeTurnEx(chessPiece, sq, turnSelector, prevPos, depthLevel);
                if (!success) continue;

                int valMinMax = 0;
                Pair<Integer, Pair<Piece, Square>> r = MinMax_SelectPiece(
                        !turnSelector, depthLevel, prevPos, tempFutureMoves);
                valMinMax = r.getKey();

                // Undo the move
                chessPiece.move(currSq);
                if (capturedPiece != null) {
                    if ((capturedPiece.getColor() == 0) && (!Bpieces.contains(capturedPiece))) {
                        Bpieces.add(capturedPiece);
                    }
                    else if ((capturedPiece.getColor() == 1) && (!Wpieces.contains(capturedPiece))) {
                        Wpieces.add(capturedPiece);
                    }
                    capturedPiece.move(sq);
                }
                cmd.update();

                if ((turnSelector) && ((valMinMax < nextMoveMinMax) || (nextMoveMinMax == 0))) {
                    // Human Player (White) Move
                    nextMoveSq = sq;
                    nextMoveMinMax = valMinMax;
                    bestFutureMoves.removeAllElements();
                    bestFutureMoves.addAll(tempFutureMoves);
                }
                if ((!turnSelector) && ((valMinMax > nextMoveMinMax) || (nextMoveMinMax == 0))) {
                    // Computer Player (Black) Move
                    nextMoveSq = sq;
                    nextMoveMinMax = valMinMax;
                    bestFutureMoves.removeAllElements();
                    bestFutureMoves.addAll(tempFutureMoves);
                }
            }
        }

        if(nextMoveSq != null){
            bestNextMove = nextMoveSq;
            futureMoves.addAll(bestFutureMoves);
            return new Pair<Integer, Square>(nextMoveMinMax, bestNextMove);
        }
        else {
            return new Pair<Integer, Square>(0, null);
        }
    }

    private boolean EvadeCheck() {

        Stack<String> tempFutureMoves = new Stack<String>();
        // Try to find best square to move the King
        Pair<Integer, Square> r = MinMax_SelectSquare(Bk, false, 0, Bk.getPositionName() + "\r\n", tempFutureMoves);
        int valMinMax = r.getKey();
        Square sq = r.getValue();

        if (!takeTurnEx(Bk, sq, false, "", 0)) {
            List<Square> kingsMoves = Bk.getLegalMoves(this);
            Iterator<Square> iterator = kingsMoves.iterator();

            // If best square is not available pick any available square
            while (iterator.hasNext()) {
                sq = iterator.next();
                if (!cmd.testMove(Bk, sq)) continue;
                if (cmd.wMoves.get(sq.hashCode()).isEmpty()) {
                    takeTurnEx(Bk, sq, false, "", 0);
                    return true;
                }
            }
        }
        else {
            return true;
        }


        return false;
    }

    private boolean takeTurnEx(Piece piece, Square sq, boolean turnSelector, String prevPos, int depthLevel) {
        String newText = "";
        boolean success = false;
        if (piece != null) {
            if (piece.getColor() == 0 && turnSelector) {
                newText = prevPos + "Black Piece on White's turn\r\n";
                return false;
            }
            else if (piece.getColor() == 1 && !turnSelector) {
                newText = prevPos + "White Piece on Black's turn\r\n";
                return false;
            }
            else {
                List<Square> legalMoves = piece.getLegalMoves(this);
                movable = cmd.getAllowableSquares(turnSelector);

                if (legalMoves.contains(sq) && movable.contains(sq)
                        && cmd.testMove(piece, sq)) {
                    sq.setDisplay(true);
                    piece.move(sq);
                    cmd.update();
                    success = true;

                    if (g.watchMoves.isSelected()) {
                        int valMinMax = MinMax_CalcVal(turnSelector);
                        newText = prevPos + piece.getPositionName();
                        newText = newText + " Level: " + Integer.toString(depthLevel);
                        newText = newText + " Val: " + Integer.toString(valMinMax) + "\r\n";
                        g.moves.setText(newText);
                        g.moves.update(g.moves.getGraphics());
                    }

                    if (cmd.blackCheckMated()) {
                        newText = newText + "Black Checkmated\r\n";
                    } else if (cmd.whiteCheckMated()) {
                        newText = newText + "White Checkmated\r\n";
                    }
                    else {
                        if (cmd.blackInCheck()) {
                            newText = newText + "Black in Check\r\n";
                        } else if (cmd.whiteInCheck()) {
                            newText = newText + "White in Check\r\n";
                        }
                    }
                }
            }
        }

        if (g.watchMoves.isSelected()) {
            this.update(this.getGraphics());
        }

        return success;
    }

    private void takeTurn(Square sq) {
        String newText = "";
        if (currPiece != null) {
            if (currPiece.getColor() == 0 && whiteTurn) {
                newText = "Black Piece on White's turn\r\n";
            }
            else if (currPiece.getColor() == 1 && !whiteTurn) {
                newText = "White Piece on Black's turn\r\n";
            }
            else {
                List<Square> legalMoves = currPiece.getLegalMoves(this);
                movable = cmd.getAllowableSquares(whiteTurn);

                if (legalMoves.contains(sq) && movable.contains(sq)
                        && cmd.testMove(currPiece, sq)) {
                    sq.setDisplay(true);
                    currPiece.move(sq);
                    cmd.update();

                    newText = currPiece.getPositionName() + "\r\n";

                    if (cmd.blackCheckMated()) {
                        currPiece = null;
                        repaint();
                        this.removeMouseListener(this);
                        this.removeMouseMotionListener(this);
                        g.checkmateOccurred(0);
                        newText = newText + "Black Checkmated\r\n";
                    } else if (cmd.whiteCheckMated()) {
                        currPiece = null;
                        repaint();
                        this.removeMouseListener(this);
                        this.removeMouseMotionListener(this);
                        g.checkmateOccurred(1);
                        newText = newText + "White Checkmated\r\n";
                    }
                    else {
                        boolean bInCheck = cmd.blackInCheck();
                        if (bInCheck) {
                            newText = newText + "Black in Check\r\n";

                            g.gameStatus.setText("Status: Computing");
                            g.buttons.update(g.buttons.getGraphics());

                            if (EvadeCheck()) {
                                currPiece = Bk;
                                whiteTurn = !whiteTurn;
                                newText = newText + "Check evaded\r\n";

                                g.gameStatus.setText("Status: Move to " + currPiece.getPositionName());
                                g.buttons.update(g.buttons.getGraphics());
                            }
                        } else if (cmd.whiteInCheck()) {
                            newText = newText + "White in Check\r\n";
                        }

                        currPiece = null;
                        whiteTurn = !whiteTurn;
                        if (!whiteTurn) {
                            // Let Computer pick the next turn
                            g.gameStatus.setText("Status: Computing");
                            g.buttons.update(g.buttons.getGraphics());

                            Stack<String> futureMoves = new Stack<String>();
                            Pair<Integer, Pair<Piece, Square>> r = MinMax_SelectPiece(false, 0, newText, futureMoves);
                            Pair<Piece, Square> m = r.getValue();
                            currPiece = m.getKey();
                            boolean success = takeTurnEx(m.getKey(), m.getValue(), whiteTurn, newText, 0);
                            whiteTurn = true; // Change the turn back to White

                            //newText = g.moves.getText();
                            newText += "Anticipated Moves:\r\n";
                            String futureMove = futureMoves.isEmpty() ? "" : futureMoves.pop();
                            while (!futureMoves.isEmpty()) {
                                newText += futureMove + "\r\n";
                                futureMove = futureMoves.pop();
                            }

                            g.gameStatus.setText("Status: Move to " + currPiece.getPositionName());
                            g.buttons.update(g.buttons.getGraphics());
                        }
                    }
                } else {
                    currPiece.getPosition().setDisplay(true);
                    currPiece = null;
                    newText = newText + "Invalid Move\r\n";
                }
            }
        }
        else {
            newText = "Null Piece\r\n";
        }

        g.moves.setText(newText);

        repaint();
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        Square sq = (Square) this.getComponentAt(new Point(e.getX(), e.getY()));

        takeTurn(sq);
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        currX = e.getX() - 24;
        currY = e.getY() - 24;

        repaint();
    }

    // Irrelevant methods, do nothing for these mouse behaviors
    @Override
    public void mouseMoved(MouseEvent e) {
    }

    @Override
    public void mouseClicked(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
    }

}