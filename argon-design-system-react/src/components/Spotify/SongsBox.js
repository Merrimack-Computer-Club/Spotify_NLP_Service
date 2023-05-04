/*!

=========================================================
* Argon Design System React - v1.1.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-design-system-react
* Copyright 2022 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-design-system-react/blob/master/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/
/*eslint-disable*/
import React from "react";
import { Link } from "react-router-dom";
// reactstrap components
import {
  Button,
  CardImg,
  NavItem,
  NavLink,
  Nav,
  Container,
  Row,
  Col,
} from "reactstrap";

// @mui
import {
  Card,
  Box,
  Stack,
  ImageList,
  ImageListItem,
  Tooltip,
} from '@mui/material';

export default function SongsBox({ songs }) {

  var audio = null;
  var url = null;

  /**
   * Play a songs audio if none is playing, if one is playing stop it.
   * @param {*} song 
   * @returns 
   */
  function playAudio(song) {
    if(audio == null) {
      if(song == undefined)
        return;
      audio = new Audio(song.url);
      audio.play();
    } else {
      audio.pause();
      audio = null;
    }
  }
  

  return (
    <div class="top-songs" style={{ borderTop: "4px solid black" }}>
      <div className="top-songs-label">
        <label className="custom-control-label" htmlFor="customRadio6">
        <h3 class="animate-charcter center-align">     Your Top Songs </h3>
        </label>
      </div>
      <Card sx={{ minWidth: '50rem', minHeight: 450, bgcolor: 'F0FAE4' }}>
        <Box sx={{ position: 'relative', pt: 1 }}>
          <Stack alignItems="center">
            <ImageList cols={7} rowHeight={95} sx={{ '&::-webkit-scrollbar': { display: 'none' }, width: '45rem', height: 400}}>
              {
              songs.map(song => (
                <Tooltip key={song.name + " " + song.artist} title={song.name + " " + song.artist} placement="top">
                  <div className={song.name} key={song.name} id={song.name} >
                    <ImageListItem onClick={() => playAudio(song)} key={song.name} sx={{bgcolor: '#f2d0d6', scale: '90%', transition: '0.5s', boxShadow: 1, borderRadius: 2, p: 0.5, '&:hover': { cursor: 'pointer',  scale: '120%', zIndex: 999 } }}>
                      <img src={song.image} alt={song.name} loading="lazy"/>
                      
                      
                    </ImageListItem>
                  </div>
                </Tooltip>
              ))
              }
            </ImageList>
          </Stack>
        </Box>
      </Card>
    </div>
  );
}

