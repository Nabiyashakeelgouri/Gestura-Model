const video = document.getElementById("video");
const result = document.getElementById("result");

let streamStarted = false;

// Start Camera
async function startCamera(){

const stream = await navigator.mediaDevices.getUserMedia({video:true});

video.srcObject = stream;

streamStarted = true;

// Start prediction loop
setInterval(captureFrame,1000);

}

// Capture frame from camera
async function captureFrame(){

if(!streamStarted) return;

const canvas = document.createElement("canvas");

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

const ctx = canvas.getContext("2d");

ctx.drawImage(video,0,0);

const image = canvas.toDataURL("image/jpeg");

try{

const response = await fetch("http://127.0.0.1:8000/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body: JSON.stringify({image:image})

});

const data = await response.json();

result.innerText = data.prediction;

}catch(error){

console.log("Backend not responding");

}

}
