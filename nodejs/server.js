'use strict';

const express = require('express');

// Constants
const PORT = 8000;
const HOST = '0.0.0.0';



// // App
const app = express();
const fs = require("fs");;
const xml2js = require("xml2js");
const FormData = require('form-data');
const axios = require('axios');


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
		if (err == null) {
			console.log(data);
			res.send(data)
			// res.status(200).send(data);
		} else {
			console.log(err);
		}
});
});


const predict_url = 'http://pytorch:8000/predict';
const imagePath = `pytorch/baboon.jpg`; //画像のパス
const file = fs.createReadStream(imagePath);

const form = new FormData();
form.append('image', file);

const config = {
	headers: {
		...form.getHeaders(),//これが大事らしい? https://qiita.com/kazu_death/items/a94ac4ae4d71928920c5 
		// 'X-AUTH-Token': auth_token,
		// 'X-API-Token': API_TOKEN,
	},
}


app.get('/predict', (req, res) => {
	axios.post(predict_url, form,config)
		.then(res => console.log(res.data)) //成功時
		.catch(err => console.log(err)); //失敗時
});


app.listen(PORT, HOST, () => {
	console.log(`Running on http://${HOST}:${PORT}`);
});