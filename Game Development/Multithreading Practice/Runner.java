
import bagel.*;
import java.util.Random;
import java.io.*;
import java.net.*;

public class Runner extends Game
{
    Label timeLabel;
    double time;

    Sprite player;

    Sprite startFlag;

    Sprite endFlag;
    Label winLabel;
    Animation idle;
    String clientName;
    int bestScore;

    BufferedReader stdIn =
        new BufferedReader(new InputStreamReader(System.in));
    Message fromServer;
    public void initialize()
    {
        try{
            System.out.println("Enter name: ");
            clientName = stdIn.readLine();
        } catch (IOException e) {
            System.err.println("IOException");
            System.exit(1);
        }
        createGroup("main");

        createGroup("labels");

        createGroup("wall");

        createGroup("platform");

        setScreenSize(1200,900);

        int screenHeight = 900;
        int screenWidth = 1200;
        int backgroundSize = 30;
        for(int i = 0; i < screenWidth/backgroundSize; i++){
            for (int j = 0; j < screenHeight/backgroundSize; j++){
                Sprite background = new Sprite();
                Animation backgroundAnimation = new Animation("images/Background/Brown.png",2,2,.5,true);
                background.setAnimation(backgroundAnimation);
                background.setSize(backgroundSize,backgroundSize);
                background.setPosition(backgroundSize*i,backgroundSize*j);
                addSpriteToGroup(background,"main");
            }
        }

        player = new Sprite();
        idle = new Animation("images/mainCharacters/virtualGuy/idle.png", 1, 11, .1, true);
        player.setAnimation(idle);
        player.setSize(50,50);
        player.setPosition(100, screenHeight-350);
        player.setPhysics( new Physics(80, 200 ,80) );
        addSpriteToGroup(player,"main");

        int wallSize = 120;
        for(int i = 0; i < screenWidth/wallSize; i++){
            Sprite wall = new Sprite();
            wall.setTexture(new Texture("images/terrain/wall.png"));
            wall.setPosition(i* wallSize,0);
            wall.setSize(wallSize,20);
            addSpriteToGroup(wall, "wall");
        }
        wallSize= 100;

        for(int i = 0; i < (screenHeight-40)/wallSize; i++){
            Sprite leftWall = new Sprite();
            Sprite rightWall = new Sprite();
            leftWall.setTexture(new Texture("images/terrain/wallLeft.png"));
            rightWall.setTexture(new Texture("images/terrain/wallRight.png"));
            leftWall.setPosition(0,(i* wallSize)+20);
            rightWall.setPosition(screenWidth-40,(i*wallSize)+20);
            leftWall.setSize(20,wallSize);
            rightWall.setSize(20,wallSize);
            addSpriteToGroup(leftWall, "wall");
            addSpriteToGroup(rightWall, "wall");
        }

        int groundSize = 100;
        for(int i = 0; i < screenWidth/groundSize; i++){
            Sprite ground = new Sprite();
            ground.setTexture(new Texture("images/terrain/ground.png"));
            ground.setSize(groundSize,150);
            ground.setPosition(i*groundSize,screenHeight-150);
            addSpriteToGroup(ground, "platform");
        }

        Animation startFlagAnimation = new Animation("images/items/checkpoints/start/start.png" ,1,17,.05,true);
        startFlag = new Sprite();
        startFlag.setAnimation(startFlagAnimation);
        startFlag.setSize(100,100);
        startFlag.setPosition(50, screenHeight-250);
        addSpriteToGroup(startFlag, "main");

        endFlag = new Sprite();
        endFlag.setTexture(new Texture("images/items/checkpoints/end/endText.png"));
        endFlag.setSize(100,100);
        endFlag.setPosition(screenWidth-250,screenHeight-250);
        addSpriteToGroup(endFlag,"main");

        time = 0;
        timeLabel = new Label();
        timeLabel.setText("Current Time: " + time);
        timeLabel.setPosition( 20, 100);
        timeLabel.setFont("Impact", 30);
        timeLabel.setColor(0.00, 0.00, 1.00);
        addSpriteToGroup( timeLabel, "labels" );

        winLabel = new Label();
        winLabel.setText("Nice job!!!");
        winLabel.setPosition( 250, 300);
        winLabel.setFont("Impact", 80);
        winLabel.setColor(0.80, 0.80, 0.80);
        winLabel.setVisible(false);
        addSpriteToGroup(winLabel, "labels");
    }

    public void update()
    {
        if(player.overlap(endFlag)){
            try (
            Socket socket = new Socket("localhost", 4444);
            ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
            ObjectInputStream in = new ObjectInputStream(socket.getInputStream());
            ) {

                out.writeObject(new Message(clientName,time));
                try{
                    fromServer = (Message) in.readObject();
                    if(fromServer.getPersonal() == fromServer.getBest()){
                        winLabel.setText("Great job, you beat the best time!!");
                    }
                    else if (fromServer.getImproved()){
                        winLabel.setText("Nice!! You beat your best score!");
                    }
                    else{
                        winLabel.setText("Try again.");
                    }
                }catch(ClassNotFoundException e){
                    System.err.println("Class not found");
                    System.exit(1);
                }

            } catch (UnknownHostException e) {
                System.err.println("Don't know about host " + "localhost");
                System.exit(1);
            } catch (IOException e) {
                System.err.println("Couldn't get I/O for the connection to " +
                    "localhost");
                System.exit(1);
            }
            winLabel.setVisible(true);
        }

        player.physics.accelerateAtAngle(90);
        if (input.isKeyPressing("A"))
            player.physics.accelerateAtAngle( 180 );

        if (input.isKeyPressing("D"))
            player.physics.accelerateAtAngle( 0 );

        if ( winLabel.visible == true )
        {
            return;
        }

        for(Sprite ground : getGroupSpriteList("platform")){
            player.preventOverlap(ground);
        }

        for(Sprite wall : getGroupSpriteList("wall")){
            player.preventOverlap(wall);
        }

        time += 1.0 / 60.0; 
        timeLabel.setText("Current Time: " + (Math.round(time*100)/100.0));

    }
}

