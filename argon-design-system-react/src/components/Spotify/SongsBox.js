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
import React, { useState } from "react";
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

  const [state, setState] = useState({
    audio: null
  })

  /**
   * Play a songs audio if none is playing, if one is playing stop it.
   * @param {*} song 
   * @returns 
   */
  function playAudio(song) {
    const state_audio = state.audio;

    if (state_audio != null) {
      if (state_audio.src === song.url) {
        if (state_audio.paused) {
          state_audio.play();
        } else {
          state_audio.pause();
        }
        return;
      } else {
        stopAudio();
      }
    }

    var audio = new Audio(song.url);
    audio.play();
    setState({ audio: audio });
  }

  /**
   * Stops the audio if it is playing
   */
  function stopAudio() {
    const state_audio = state.audio;

    if (state_audio != null) {
      state_audio.pause();
      state_audio.currentTime = 0;
      setState({ audio: null });
    }
  }


  return (
    <div class="top-songs" onBlur={() => stopAudio()}>
      <Card sx={{ minWidth: '50rem', minHeight: 450, bgcolor: 'F0FAE4' }}>
        <Box sx={{ position: 'relative', pt: 1 }}>
          <Stack alignItems="center">
            <ImageList cols={7} rowHeight={95} sx={{ '&::-webkit-scrollbar': { display: 'none' }, width: '45rem', height: 400 }}>
              {
                songs.map(song => (
                  <Tooltip key={song.name + " " + song.artist} title={song.name + " " + song.artist} placement="top">
                    <div className={song.name} key={song.name} id={song.name} >
                      <ImageListItem onClick={() => playAudio(song)} key={song.name} sx={{ bgcolor: '#f2d0d6', scale: '90%', transition: '0.5s', boxShadow: 1, borderRadius: 2, p: 0.5, '&:hover': { cursor: 'pointer', scale: '120%', zIndex: 999 } }}>
                        <img src={song.image} alt={song.name} loading="lazy" />

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

