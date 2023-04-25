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
  Card,
  CardImg,
  NavItem,
  NavLink,
  Nav,
  Container,
  Row,
  Col,
  UncontrolledTooltip
} from "reactstrap";

// @mui
import {
  Box,
  Stack,
  ImageList,
  ImageListItem,
} from '@mui/material';


export default function SongsBox({ songs }) {

  return (
    <Card sx={{ minWidth: '50rem', minHeight: 350, bgcolor: 'black' }}>
      <Box sx={{ position: 'relative', pt: 1 }}>
        <Stack alignItems="center">
          <ImageList cols={7} rowHeight={95} sx={{ '&::-webkit-scrollbar': { display: 'none' }, width: '45rem', height: 300 }}>
            {
            songs.map(song => (
              <div key={song.name}>
                <ImageListItem key={song.name} sx={{ scale: '90%', transition: '0.5s', '&:hover': { cursor: 'pointer',  scale: '120%', zIndex: 999 } }}>
                  <img src={song.image} alt={song.name} loading="lazy"/>
                </ImageListItem>
              </div>
            ))
            }
          </ImageList>
        </Stack>
      </Box>
    </Card>
  );
}

