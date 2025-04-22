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
    if ($_POST["change"]== "add") {
        $sql = "INSERT INTO IPAddress VALUES('" . $_POST["ip"] . "', " . $_POST["id"] . ", '" . $_POST["reason"] . "');";
       
        if ($conn->query($sql) === TRUE) {
            echo "IP added successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
        $sql = "INSERT INTO Comment VALUES(NULL, 'added IP','" . $_POST["handler"] . "', " . $_POST["id"] . ");";


        if ($conn->query($sql) === TRUE) {
            echo "Comment created successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
    if ($_POST["change"]  == "delete"){


        $sql = "DELETE FROM IPADDRESS WHERE email ='" . $_POST["email"] . "';";


        if ($conn->query($sql) === TRUE) {
            echo "IP deleted successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
         $sql = "INSERT INTO Comment VALUES(NULL, 'removed IP','" . $_POST["handler"] . "', " . $_POST["id"] . ");";


        if ($conn->query($sql) === TRUE) {
            echo "Comment created successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
   
    }
    $conn->close();
?>
