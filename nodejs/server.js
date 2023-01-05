'use strict';

const express = require('express');

// Constants
const PORT = 8000;
const HOST = '0.0.0.0';



// // App
const app = express();
const fs = require("fs");;
const xml2js = require("xml2js");


const request = require('request');
var URL = 'http://zip.cgis.biz/xml/zip.php';


app.get('/', (req, res) => {
  let zn = req.query.zn
  if (!req.query.zn){
    zn = "0640805"
  }

  console.log(req.query.zn)
  request.get({
    uri: URL,
    // headers: {'Content-type': 'application/json'},
    qs: {
      zn:zn
    },

    // json: true
}, function(err, req, data){
    console.log(zn),
    console.log(data);
    xml2js.parseString(data, function (err, result) {
      if (err) {
        console.log(err.message)
      } else {
        console.log(result)
        res.send(result);
       }
    });

});
});


app.get('/fastapi', (req, res) => {


  console.log(req.query.zn)
  request.get({
    uri: 'http://fastapi:8000/test_api', // http入れなきゃだめみたい
    // headers: {'Content-type': 'application/json'},
    json: true

}, function(err, req, data){
    // console.error(err);
    // console.log(req);
    // res.send(data);
    // console.log(err)
    // console.log("test")
    if (err == null) {
      console.log(data);
      res.send(data)
      // res.status(200).send(data);
    } else {
      console.log(err);
    }
});
});



app.listen(PORT, HOST, () => {
  console.log(`Running on http://${HOST}:${PORT}`);
});