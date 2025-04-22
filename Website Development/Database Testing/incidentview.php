<!DOCTYPE html>
<html>
<head>
    <title>Incident</title>
    <style type="text/css">
        table, th, td {border: 1px solid black}
    </style>
    <link rel="stylesheet" href="./styles.css">
</head>
<body>
    <p><i><?php  echo "PHP has been installed successfully!"; ?></i></p>
 
    <p><?php
        $servername = "localhost";
        $username = "root"; // Mysql username
        $password = "1234"; // Mysql Password
 
        $dbname = "tracker";   // database name
         
        // Create connection
        // MySQLi is Object-Oriented method
        $conn = new mysqli($servername, $username, $password, $dbname);
         
        // Check connection
        if ($conn->connect_error) {
            die("Connection failed: " . $conn->connect_error ."<br>");
        }
        echo "<i>DB Connected successfully...</i>";
        ?>
    </p>
 
 
    <table>
        <tr>
            <th>Incident ID</th>
            <th>Type</th>
            <th>Date</th>
            <th>State</th>
        </tr>
        <?php
            $sql = "SELECT incidentNum, type, day, state FROM Incident WHERE IncidentNum=" . $_POST["id"] . ";";
            $result = $conn->query($sql);
 
            // Make sure the relation is not empty
            if($result -> num_rows > 0){
                while($row = $result -> fetch_assoc()) {
                    echo "<tr>
                                <td>" . $row["incidentNum"] . "</td>
                                <td>" . $row["type"] . "</td>
                                <td>" . $row["day"] . "</td>
                                <td>" . $row["state"] . "</td>
                         </tr>" ;
                }
            } else {
                // empty list
                echo "<tr> 0 results </tr>";
            }
        ?>
    </table>
   
    <table>
        <tr>
            <th>Email</th>
            <th>Last Name</th>
            <th>First Name</th>
            <th>Job</th>
            <th>Reason</th>
        </tr>
        <?php
            $sql = "SELECT * FROM People WHERE IncidentNum=" . $_POST["id"] . ";";
            $result = $conn->query($sql);
 
            if($result -> num_rows > 0){
                while($row = $result -> fetch_assoc()) {
                    echo "<tr>
                                <td>" . $row["email"] . "</td>
                                <td>" . $row["lastName"] . "</td>
                                <td>" . $row["firstName"] . "</td>
                                <td>" . $row["job"] . "</td>
                                <td>" . $row["reason"] . "</td>
                         </tr>" ;
                }
            } else {
                // empty list
                echo "<tr> No people involved </tr>";
            }
        ?>
 
    <table>
        <tr>
            <th>IP</th>
            <th>Reason</th>
        </tr>
        <?php
            $sql = "SELECT * FROM IPAddress WHERE IncidentNum=" . $_POST["id"] . ";";
            $result = $conn->query($sql);
 
            if($result -> num_rows > 0){
                while($row = $result -> fetch_assoc()) {
                    echo "<tr>
                                <td>" . $row["ip"] . "</td>
                                <td>" . $row["reason"] . "</td>
                         </tr>" ;
                }
            } else {
                // empty list
                echo "<tr> No IP's involved </tr>";
            }
        ?>
    </table>


 <?php
        echo "<br> Comments <br>";
        $sql = "SELECT j.text, j.handler, i.day
        FROM Incident i
        INNER JOIN Comment j ON i.incidentNum=" . $_POST["id"] . "
      AND i.incidentNum = j.incidentNum
        ORDER BY i.day ASC;";
       
        $result = $conn->query($sql);
        while($row = $result -> fetch_assoc()) {
            echo " - " . $row["text"] . " (Written by " . $row["handler"] . ")<br>";
        }
    ?>




    <!-- change state -->
    <?php
        echo "<p>
               <form method=\"post\" action= \"./statechanger.php\">
                    <label for=\"state\">New state: </label>
                    <input type = \"text\" name=\"state\"><br>
                    <label for=\"id\">Incident ID: </label>
                    <input type = \"number\" name=\"id\" value=" . $_POST["id"] . ">
                    <label for=\"handler\">Handler: </label>
                    <input type = \"text\" name=\"handler\"><br>     <br><br>
                    <input type =\"submit\" value=\"Submit\">
                </form></p>";
    ?>
   


    <!-- add comment -->  
    <?php
        echo "<p>
               <form method=\"post\" action= \"./addcomment.php\">
                    <label for=\"txt\">Comment: </label>
                    <input type = \"text\" name=\"txt\"><br>


                    <label for=\"id\">Incident ID: </label>
                    <input type = \"number\" name=\"id\" value=" . $_POST["id"] . "><br>
                   
                    <label for=\"handler\">Handler: </label>
                    <input type = \"text\" name=\"handler\">     <br><br>


                    <input type =\"submit\" value=\"Submit\">
                </form></p>";
    ?>



  <!-- change add/remove people -->  
    <?php
        echo "<p>
               <form method=\"post\" action= \"./people.php\">
                    <input type=\"radio\"  name=\"change\" value=\"add\">Add<br>
                    <input type=\"radio\"  name=\"change\" value=\"delete\">Delete<br>


                    <label for=\"email\">Email: </label>
                    <input type = \"text\" name=\"email\"><br>


                    <label for=\"fName\">First Name: </label>
                    <input type = \"text\" name=\"fName\"><br>
                   
                    <label for=\"lName\">Last Name: </label>
                    <input type = \"text\" name=\"lName\"><br>


                    <label for=\"job\">Job: </label>
                    <input type = \"text\" name=\"job\"><br>


                    <label for=\"reason\">Reason: </label>
                    <input type = \"text\" name=\"reason\"><br>


                    <label for=\"id\">Incident ID: </label>
                    <input type = \"number\" name=\"id\" value=" . $_POST["id"] . "><br>
                   
                    <label for=\"handler\">Handler: </label>
                    <input type = \"text\" name=\"handler\">     <br><br>


                    <input type =\"submit\" value=\"Submit\">
                </form></p>";
    ?>

 <!-- change add/remove ips -->  
    <?php
        echo "<p>
               <form method=\"post\" action= \"./ip.php\">
                    <input type=\"radio\"  name=\"change\" value=\"add\">Add<br>
                        <input type=\"radio\"  name=\"change\" value=\"delete\">Delete<br>


                    <label for=\"ip\">ip: </label>
                    <input type = \"text\" name=\"ip\"><br>


                    <label for=\"reason\">Reason: </label>
                    <input type = \"text\" name=\"reason\"><br>


                    <label for=\"id\">Incident ID: </label>
                    <input type = \"number\" name=\"id\" value=" . $_POST["id"] . "><br>
                   
                    <label for=\"handler\">Handler: </label>
                    <input type = \"text\" name=\"handler\">     <br><br>


                    <input type =\"submit\" value=\"Submit\">
                </form></p>";
    ?>


    <p><?php
        $conn->close();
 
        echo "<i>Searching Completed. <br>...DB Disconnect. Done.</i>";
    ?></p>


   
</body>
</html>
