
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.*;


public class GameWindow {
    private JFrame gameWindow;
    
    public Clock blackClock;
    public Clock whiteClock;
    
    private Timer timer;
    
    private Board board;
    
    public JTextArea moves;
    public JCheckBox watchMoves;
    public JLabel gameStatus;
    public JPanel buttons;
    public JTextField depth;

    public GameWindow(String blackName, String whiteName, int hh, 
            int mm, int ss) {
        
        blackClock = new Clock(hh, ss, mm);
        whiteClock = new Clock(hh, ss, mm);
        
        gameWindow = new JFrame("Chess");
        

        try {
            Image whiteImg = ImageIO.read(getClass().getResource("wp.png"));
            gameWindow.setIconImage(whiteImg);
        } catch (Exception e) {
            System.out.println("Game file wp.png not found");
        }

        gameWindow.setLocation(100, 100);
        
        
        gameWindow.setLayout(new BorderLayout(20,20));
       
        // Game Data window
        JPanel gameData = gameDataPanel(blackName, whiteName, hh, mm, ss);
        gameData.setSize(gameData.getPreferredSize());
        gameWindow.add(gameData, BorderLayout.NORTH);
        
        this.board = new Board(this);
        
        gameWindow.add(board, BorderLayout.CENTER);
        
        gameWindow.add(buttons(), BorderLayout.SOUTH);
        
        gameWindow.setMinimumSize(gameWindow.getPreferredSize());
        gameWindow.setSize(gameWindow.getPreferredSize());
        gameWindow.setResizable(true);
        
        gameWindow.pack();
        gameWindow.setVisible(true);
        gameWindow.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }
    
// Helper function to create data panel
    
    private JPanel gameDataPanel(final String bn, final String wn, 
            final int hh, final int mm, final int ss) {
        
        JPanel gameData = new JPanel();
        gameData.setLayout(new GridLayout(3,2,0,0));
        
        
        // PLAYER NAMES
        
        JLabel w = new JLabel(wn);
        JLabel b = new JLabel(bn);
        
        w.setHorizontalAlignment(JLabel.CENTER);
        w.setVerticalAlignment(JLabel.CENTER);
        b.setHorizontalAlignment(JLabel.CENTER);
        b.setVerticalAlignment(JLabel.CENTER);
        
        w.setSize(w.getMinimumSize());
        b.setSize(b.getMinimumSize());
        
        gameData.add(w);
        gameData.add(b);
        
        // CLOCKS
        
        final JLabel bTime = new JLabel(blackClock.getTime());
        final JLabel wTime = new JLabel(whiteClock.getTime());
        
        bTime.setHorizontalAlignment(JLabel.CENTER);
        bTime.setVerticalAlignment(JLabel.CENTER);
        wTime.setHorizontalAlignment(JLabel.CENTER);
        wTime.setVerticalAlignment(JLabel.CENTER);
        
        if (!(hh == 0 && mm == 0 && ss == 0)) {
            timer = new Timer(1000, null);
            timer.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent e) {
                    boolean turn = board.getTurn();
                    
                    if (turn) {
                        whiteClock.decr();
                        wTime.setText(whiteClock.getTime());
                        
                        if (whiteClock.outOfTime()) {
                            timer.stop();
                            int n = JOptionPane.showConfirmDialog(
                                    gameWindow,
                                    bn + " wins by time! Play a new game? \n" +
                                    "Choosing \"No\" quits the game.",
                                    bn + " wins!",
                                    JOptionPane.YES_NO_OPTION);
                            
                            if (n == JOptionPane.YES_OPTION) {
                                new GameWindow(bn, wn, hh, mm, ss);
                                gameWindow.dispose();
                            } else gameWindow.dispose();
                        }
                    } else {
                        blackClock.decr();
                        bTime.setText(blackClock.getTime());
                        
                        if (blackClock.outOfTime()) {
                            timer.stop();
                            int n = JOptionPane.showConfirmDialog(
                                    gameWindow,
                                    wn + " wins by time! Play a new game? \n" +
                                    "Choosing \"No\" quits the game.",
                                    wn + " wins!",
                                    JOptionPane.YES_NO_OPTION);
                            
                            if (n == JOptionPane.YES_OPTION) {
                                new GameWindow(bn, wn, hh, mm, ss);
                                gameWindow.dispose();
                            } else gameWindow.dispose();
                        }
                    }
                }
            });
            timer.start();
        } else {
            wTime.setText("Untimed game");
            bTime.setText("Untimed game");
        }
        
        gameData.add(wTime);
        gameData.add(bTime);
        
        gameData.setPreferredSize(gameData.getMinimumSize());
        
        return gameData;
    }
    
    private JPanel buttons() {
        buttons = new JPanel();
        buttons.setLayout(new GridLayout(1, 2, 10, 0));

        final JButton quit = new JButton("Quit");
        
        quit.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int n = JOptionPane.showConfirmDialog(
                        gameWindow,
                        "Are you sure you want to quit?",
                        "Confirm quit", JOptionPane.YES_NO_OPTION);
                
                if (n == JOptionPane.YES_OPTION) {
                    if (timer != null) timer.stop();
                    gameWindow.dispose();
                }
            }
          });
        
        final JButton nGame = new JButton("New game");
        
        nGame.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int n = JOptionPane.showConfirmDialog(
                        gameWindow,
                        "Are you sure you want to begin a new game?",
                        "Confirm new game", JOptionPane.YES_NO_OPTION);
                
                if (n == JOptionPane.YES_OPTION) {
                    SwingUtilities.invokeLater(new StartMenu());
                    gameWindow.dispose();
                }
            }
          });
        
        final JButton instr = new JButton("How to play");
        
        instr.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                JOptionPane.showMessageDialog(gameWindow,
                        "Move the chess pieces on the board by clicking\n"
                        + "and dragging. The game will watch out for illegal\n"
                        + "moves. You can win either by your opponent running\n"
                        + "out of time or by checkmating your opponent.\n"
                        + "\nGood luck, hope you enjoy the game!",
                        "How to play",
                        JOptionPane.PLAIN_MESSAGE);
            }
          });

        moves = new JTextArea("", 4, 5);

        JScrollPane scroll = new JScrollPane (moves,
                JScrollPane.VERTICAL_SCROLLBAR_ALWAYS, JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);

        watchMoves = new JCheckBox();
        watchMoves.setText("Watch Moves");
        watchMoves.setSelected(false);

        gameStatus = new JLabel("Status: ");

        depth = new JTextField("Evaluation Depth: 2");

        JPanel actButtons = new JPanel();
        actButtons.setLayout(new GridLayout(3, 2, 10, 10));
        actButtons.add(instr);
        actButtons.add(watchMoves);
        actButtons.add(nGame);
        actButtons.add(gameStatus);
        actButtons.add(quit);
        actButtons.add(depth);

        buttons.add(actButtons);
        buttons.add(scroll);

        buttons.setPreferredSize(new Dimension(600, 100));
        
        return buttons;
    }

    public void showMessage(String msg) {
        int n = JOptionPane.showConfirmDialog(
                gameWindow, msg, "Message", JOptionPane.OK_OPTION);
    }

    public void checkmateOccurred (int c) {
        if (c == 0) {
            if (timer != null) timer.stop();
            int n = JOptionPane.showConfirmDialog(
                    gameWindow,
                    "White wins by checkmate! Set up a new game? \n" +
                    "Choosing \"No\" lets you look at the final situation.",
                    "White wins!",
                    JOptionPane.YES_NO_OPTION);
            
            if (n == JOptionPane.YES_OPTION) {
                SwingUtilities.invokeLater(new StartMenu());
                gameWindow.dispose();
            }
        } else {
            if (timer != null) timer.stop();
            int n = JOptionPane.showConfirmDialog(
                    gameWindow,
                    "Black wins by checkmate! Set up a new game? \n" +
                    "Choosing \"No\" lets you look at the final situation.",
                    "Black wins!",
                    JOptionPane.YES_NO_OPTION);
            
            if (n == JOptionPane.YES_OPTION) {
                SwingUtilities.invokeLater(new StartMenu());
                gameWindow.dispose();
            }
        }
    }
}
