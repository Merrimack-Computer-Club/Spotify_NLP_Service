import React from "react";


const clientId = 'e15a2433aba3444fbba3d0e7c07b9bd3';
const redirectUri = 'http://localhost:3000/data';

export default function log() {
  console.log("Default function from SpotifyOath");
}

export async function authorize() {
    let codeVerifier = generateRandomString(128);
    localStorage.setItem('code_verifier', codeVerifier);


    generateCodeChallenge(codeVerifier).then(codeChallenge => {

        let state = generateRandomString(16);
        let scope = 'user-read-private user-read-email user-read-playback-state user-modify-playback-state user-read-recently-played user-top-read user-read-private user-read-email user-top-read user-library-read ';

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

  // Get the profile and if it is not a status 200 then get a new access token
  //const test_resp = await getProfile();
  //if(test_resp.status == 200)
  // return;

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
                window.location = 'localhost:3000';
                return response.json();
            })
            .then(data => {
                localStorage.setItem('access_token', data.access_token);
            })
            .catch(error => {
                console.error('Error:', error);
        });
}

export async function getTopSongs() {
  let accessToken = localStorage.getItem('access_token');

  let body = new URLSearchParams({
    time_range: 'medium_term',
    limit: '50',
    offset: '0'
  });

  const response = await fetch(`https://api.spotify.com/v1/me/top/tracks?` + body, {
      method: 'GET',
      headers: {
        Authorization: 'Bearer ' + accessToken
      },
  });

  const data = await response.json();
  console.log(data);

  // Get all of the song names
  let names = data.items.map(n => n.name + " | " + n.external_ids.isrc);
  console.log(names)

  return data;

}

export async function getProfile() {
    let accessToken = localStorage.getItem('access_token');

    const response = await fetch('https://api.spotify.com/v1/me', {
      headers: {
        Authorization: 'Bearer ' + accessToken
      }
    });
  
    const data = await response.json();

    // If there was an error getting the profile, authorize again with the code. 
    //if(data.error !== undefined) {
    //    const resp = await getResponse();
    //    return;
    //}


    return response;
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

/**
 * Gets the top songs
 * @returns a pair of <song_name, image_url>
 */
export async function getTopSongsInfo() {


  const data = await getTopSongs();
  
  const song_objs = [];
  // Get all of the song names
  data.items.forEach(item => {

    let name = "";
    if(item.name != undefined)
      name = item.name;

    let img_url = "";
    if(item.album.images[0].url != undefined)
      img_url = item.album.images[0].url;

    let artist = "| ";
    if(item.artists != undefined)
      item.artists.forEach(p => artist += p.name + " ")

    let url = "";
    if(item.preview_url != undefined)
      url = item.preview_url;
  
    let add = true;
    if(song_objs.forEach(song => {
      if(song.name === name)
        add = false;
    }));
      
    if(add)
      song_objs.push(new song(name, img_url, artist, url));
  });

  return song_objs;
}
  
// Constructs a song from a song, image pair.
function song(name, image, artist, url) {
  this.name = name;
  this.image = image;
  this.artist = artist;
  this.url = url;
}

