
function validateEmail(emailField) {
    var reg = /^([A-Za-z0-9_\-\.])+\@([A-Za-z0-9_\-\.])+\.([A-Za-z]{2,4})$/;
    console.log(emailField)
    if (reg.test(emailField) == false) {
        return false;
    }
    return true;

}

function doPostStart(){
    const url = 'http://localhost:5000/start'
    var time = document.getElementById("times").value;
    var doll = document.getElementById("doll").value;
    var email = document.getElementById("email").value;
    var algo = document.getElementById("algo").value;
    console.log(validateEmail(email))
    if (!validateEmail(email) && email != ""){
        window.alert("Email is not valid, please try again");
        return;
    }
    stopButton = document.getElementById("stop");
    stopButton.disabled = false;
    startButton = document.getElementById("start");
    startButton.disabled = true;
    let reqData = {
        times: time,
        doll: doll,
        email: email,
        algo: algo

    }
    fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(reqData),
        mode: "cors"
    })
    .then(res => {
        console.log(res);
    })
    .catch(error => {
        window.alert(error);
        console.log("Error:", error)
    });
}

function doPostStop(){
    stopButton = document.getElementById("stop");
    stopButton.disabled = true;
    startButton = document.getElementById("start");
    startButton.disabled = false;
    const url = 'http://localhost:5000/stop'

    fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        mode: "cors"
    })
    .then(res => {
        console.log(res);
    })
    .catch(error => console.log("Error:", error));
}