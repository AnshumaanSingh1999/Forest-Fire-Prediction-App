<?php
error_reporting(E_ERROR);
?>

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="css/style.css">
    <title>Forest Fire</title>
</head>
<body>
        <h1>Enter the details:</h1>
        <form action="index.php" method="post">


        <div>
                <label for="name">Temprature:</label>
                <input type="integer" name="t" required="required"/>
            </div>
            <br>
            
            <div>
                <label for="name">Oxygen:</label>
                <input type="integer" name="o" required="required" />
            </div>
            <br>

            

            <div>
                <label for="name">Humidity:</label>
                <input type="integer" name="h" required="required"/>
            </div>
            <br>
            <button type="submit">Submit</button>
        </form>
        <br>
<?php
    $t = $_POST['t'];
    $o = $_POST['o'];
    $h = $_POST['h'];
    $command = escapeshellcmd("python script.py $t $o $h");
    $output = exec($command);
    echo $output;
?>

</body>
</html>
