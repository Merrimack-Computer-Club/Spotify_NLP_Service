import React from "react";


const clientId = 'e15a2433aba3444fbba3d0e7c07b9bd3';
const redirectUri = 'http://localhost:3000/login-page';

export default function log() {
  console.log("Default function from SpotifyOath");
}

export async function authorize() {
    let codeVerifier = generateRandomString(128);
    localStorage.setItem('code_verifier', codeVerifier);


    generateCodeChallenge(codeVerifier).then(codeChallenge => {

        let state = generateRandomString(16);
        let scope = 'user-read-private user-read-email user-top-read user-library-read ';

        let args = new URLSearchParams({
            response_type: 'code',
            client_id: clientId,
            scope: scope,
            redirect_uri: redirectUri,
            state: state,
            code_challenge_method: 'S256',
            code_challenge: codeChallenge
        });

        window.location = 'https://accounts.spotify.com/authorize?' + args;
    });
}

export async function getResponse() {

  const urlParams = new URLSearchParams(window.location.search);
  let code = urlParams.get('code');

  let codeVerifier = localStorage.getItem('code_verifier');

  let body = new URLSearchParams({
    grant_type: 'authorization_code',
    code: code,
    redirect_uri: redirectUri,
    client_id: clientId,
    code_verifier: codeVerifier
  });

  const response = fetch('https://accounts.spotify.com/api/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: body
            })
            .then(response => {
                if (!response.ok) {
                throw new Error('HTTP status ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                localStorage.setItem('access_token', data.access_token);
            })
            .catch(error => {
                console.error('Error:', error);
        });
}

export async function getProfile() {
    let accessToken = localStorage.getItem('access_token');
  
    console.log(`Access Token ${accessToken}`);

    const response = await fetch('https://api.spotify.com/v1/me', {
      headers: {
        Authorization: 'Bearer ' + accessToken
      }
    });
  
    const data = await response.json();

    console.log(data);
}

/**
 * Generates a challenge based off of the verifier (random string or the IV)
 * @param {*} codeVerifier 
 * @returns 
 */
async function generateCodeChallenge(codeVerifier) {
  function base64encode(string) {
    return btoa(String.fromCharCode.apply(null, new Uint8Array(string)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }

  console.log("error?");

  const encoder = new TextEncoder();
  const data = encoder.encode(codeVerifier);
  const digest = await window.crypto.subtle.digest('SHA-256', data);

  return base64encode(digest);
}



/**
 * Constructs an IV based off of the length
 * @param {} length 
 * @returns 
 */
function generateRandomString(length) {
    let text = '';
    let possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  
    for (let i = 0; i < length; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }

