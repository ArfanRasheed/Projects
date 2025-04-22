<?php
    $servername = "localhost";
    $username = "root"; // Mysql username
    $password = "1234"; // Mysql Password
    $dbname = "tracker";   // database name
     
    $conn = new mysqli($servername, $username, $password, $dbname);
        // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error ."<br>");
    }
   
    $sql = "UPDATE Incident SET state ='" . $_POST["state"] . "' WHERE incidentNum = " . $_POST["id"] . ";";
 
    if ($conn->query($sql) === TRUE) {
        echo "Incident updated<br>";
      } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
     }


     $sql = "INSERT INTO Comment VALUES(NULL, 'State change to " . $_POST["state"] . "', '" . $_POST["handler"] . "', " . $_POST["id"] . ");";


     if ($conn->query($sql) === TRUE) {
        echo "Comment created successfully<br>";
      } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
     }
 
    $conn->close();
?>
