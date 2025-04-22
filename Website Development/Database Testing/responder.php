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
   
    $sql = "INSERT INTO Incident VALUES(NULL, '" . $_POST["type"] . "', '" . $_POST["date"] . "', '" . $_POST["state"] . "');";
 
    if ($conn->query($sql) === TRUE) {
        echo "New incident created successfully";
      } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
     }
 
    $conn->close();
?>
