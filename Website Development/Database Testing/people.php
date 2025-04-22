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
    if ($_POST["change"] == "add") {
        $sql = "INSERT INTO People VALUES('" . $_POST["email"] . "', '" . $_POST["lName"] . "', '" . $_POST["fName"] . "', '" . $_POST["job"] . "', " . $_POST["id"] . ", '" . $_POST["reason"] . "');";
       
        if ($conn->query($sql) === TRUE) {
            echo "Person added successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
        $sql = "INSERT INTO Comment VALUES(NULL, 'added person','" . $_POST["handler"] . "', " . $_POST["id"] . ");";


        if ($conn->query($sql) === TRUE) {
            echo "Comment created successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
    if ($_POST["change"]  == "delete"){


        $sql = "DELETE FROM PEOPLE WHERE email ='" . $_POST["email"] . "';";


        if ($conn->query($sql) === TRUE) {
            echo "Person deleted successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
         $sql = "INSERT INTO Comment VALUES(NULL, 'removed person','"  . $_POST["handler"] . "', " . $_POST["id"] . ");";


        if ($conn->query($sql) === TRUE) {
            echo "Comment created successfully<br>";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
   
    }
    $conn->close();
?>
