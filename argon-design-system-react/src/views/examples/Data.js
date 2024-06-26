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

import React from "react";
import './style.css';

// reactstrap components to choose from
import {
  Button,
  CardHeader,
  CardBody,
  FormGroup,
  Form,
  Input,
  InputGroupAddon,
  InputGroupText,
  InputGroup,
  Container,
  Row,
  Col,
  Dropdown,
  DropdownToggle,
  DropdownMenu,
  DropdownItem
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

// core components
import DemoNavbar from "components/Navbars/DemoNavbar.js";
import SimpleFooter from "components/Footers/SimpleFooter.js";
import SongsBox from "components/Spotify/SongsBox.js"
import { getResponse, getProfile, getTopSongs, getTopSongsInfo, getTopSongsData, getSingleSongInfo, getSingleSongData } from "util/SpotifyOath.js";
import { Alert } from "reactstrap";
import PieChart from "components/Charts/PieChart.js"
import TreeChart from "components/Charts/TreeChart.js"

// List of emotion to select from dropdown
const emotions = ["Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion", "Curiosity", "Desire", "Disappointment", 'Disapproval', 'Disgust', 'Embarrassment', "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism", "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"];
// List of time ranges to select from dropdown
const timeframe = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

const host = process.env.REACT_APP_SERVER;
const port = process.env.REACT_APP_SERVER_PORT;

// Class extending react components
class Data extends React.Component {

  // Contructor
  constructor(props) {
    super(props);
    // Intitialzie dropdown and radiobutton components
    this.state = {
      selectedRadio: "",
      dropdownOpen_Graph_TimeFrame: false,
      dropdownOpen_Graph_SongRange: false,
      dropdownOpen_Song_Emotion: false,
      selectedVal_TimeFrame: null,
      selectedVal_SongRange: null,
      selectedVal_Emotion: null,
      selectedTitle_TimeFrame: "",
      selectedTitle_SongRange: "",
      selectedTitle_Emotion: "",
      base64_encoded_gimage: "",
      isButtonClicked: false,
      data: undefined,
      tree_data: undefined
    };


  }

  // Make sure webpage starts at top
  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;

    // Get the response code from SpotifyOath
    getResponse().then(() => {
      console.log("profile");
      getProfile();
      //getTopSongs();
      //getTopSongsData();
    });



  }

  /*
  This method is responsible for contacting the server
  to construct a graph for a single song.
  */
  sendSingleSongGraphInput = async () => {
    this.setState({ isButtonClicked: true });
    this.setState({ base64_encoded_gimage: undefined });

    // Load the top songs.
    getSingleSongInfo(this.state.selected_SongName).then(songs => this.setState({ songs }));

    // Fetch for the Data Analysis and Graph Image encoded
    console.log('Sending to http://' + host + ':' + port + '/api/emotions/post/list');
    
    const url = port == null ? 'https://' + host + '/api/emotions/post/list' : 'http://' + host + ':' + port + '/api/emotions/post/list';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: await getSingleSongData(this.state.selected_SongName)
    }).catch(error => console.error(error));

    const data = await response.json();
    const probs = data['probabilities'];
    console.log(probs)

    var data_dict = {};
    for(let i = 0; i < probs.length; i++) {
      let emotion = probs[i]['emotion'];
      if(emotion in data_dict) {
        data_dict[emotion] = data_dict[emotion]+1;
      } else {
        data_dict[emotion] = 0;
      }
    }

    const tree_result = [["Emotion", "Emotion", "Count"], ["Emotions", null, 0]];
    const result = [["Emotion", "Count"]];
    // Iterate over each key-value pair in the object
    for (const key in data_dict) {
      // Push an array containing key and value as a sub-array into the result array
      result.push([key, data_dict[key]]);
      tree_result.push([key, "Emotions", data_dict[key]]);
    }

    console.log(result);
    console.log(tree_result);

    this.setState({ data: result });
    this.setState({ tree_data: tree_result })
  }

  /*
  This method is responsible for creating 
  post requests of the graph data. Graph data
  contains info on Time Frame & Song Range the
  user has selected from the dropdown menus.
  */
  sendGraphInput = async () => {
    this.setState({ data: undefined });

    // Load the top songs.
    getTopSongsInfo(this.state.selectedVal_TimeFrame, this.state.selectedVal_SongRange).then(songs => this.setState({ songs }));

    // Fetch for the Data Analysis and Graph Image encoded
    const url = port == null ? 'https://' + host + '/api/emotions/post/list' : 'http://' + host + ':' + port + '/api/emotions/post/list';
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: await getTopSongsData(this.state.selectedVal_TimeFrame, this.state.selectedVal_SongRange)
    }).catch(error => console.error(error));

    const data = await response.json();
    const probs = data['probabilities'];
    console.log(probs)

    var data_dict = {};
    for(let i = 0; i < probs.length; i++) {
      let emotion = probs[i]['emotion'];
      if(emotion in data_dict) {
        data_dict[emotion] = data_dict[emotion]+1;
      } else {
        data_dict[emotion] = 0;
      }
    }

    const tree_result = [["Emotion", "Emotion", "Count"], ["Emotions", null, 0]];
    const result = [["Emotion", "Count"]];
    // Iterate over each key-value pair in the object
    for (const key in data_dict) {
      // Push an array containing key and value as a sub-array into the result array
      result.push([key, data_dict[key]]);
      tree_result.push([key, "Emotions", data_dict[key]]);
    }

    console.log(result);
    console.log(tree_result);

    this.setState({ data: result });
    this.setState({ tree_data: tree_result })
  }

  /*
  Handle dropdown methods work with
  user selection from the drop down menus:
  Getting the selected val & updating the dropdown menu
  title with the selected values
  */
  handleDropdownSelect_TimeFrame(value) {
    this.setState({ selectedVal_TimeFrame: value, selectedTitle_TimeFrame: value });
  }
  handleDropdownSelect_SongRange(value) {
    this.setState({ selectedVal_SongRange: value, selectedTitle_SongRange: value });
  }

  handleDropdownSelect_Emotion(value) {
    this.setState({ selectedVal_Emotion: value, selectedTitle_Emotion: value });
  }

  // Handles with radio button selection, contains value of selected radio button
  handleRadioChange = (event) => {
    this.setState({
      selectedRadio: event.target.value,
    });
  };

  // Handles text change from the Input Field Box (single song input)
  handleSingleSongInputChange = (event) => {
    this.setState({
      selected_SongName: event.target.value,
    });
  };

  // Drop down menu stuff
  toggleDropdown_TimeFrame = () => {
    this.setState({
      dropdownOpen_Graph_TimeFrame: !this.state.dropdownOpen_Graph_TimeFrame,
    });
  };
  toggleDropdown_SongRange = () => {
    this.setState({
      dropdownOpen_Graph_SongRange: !this.state.dropdownOpen_Graph_SongRange,
    });
  };
  toggleDropdown_Emotion = () => {
    this.setState({
      dropdownOpen_Song_Emotion: !this.state.dropdownOpen_Song_Emotion,
    });
  };

  // HTML CODE
  render() {
    // Constructor
    const {
      selectedRadio,
      dropdownOpen_Graph_TimeFrame,
      dropdownOpen_Graph_SongRange,
      dropdownOpen_Song_Emotion,
      selectedVal_TimeFrame,
      selectedTitle_TimeFrame,
      selectedVal_SongRange,
      selectedTitle_SongRange,
      selectedTitle_Emotion,
      selectedVal_Emotion
    } = this.state;
    return (
      <>
        <DemoNavbar /> {/* Pre built nav bar*/}
        <head>
          {/* Linked CSS file*/}
          <link rel="stylesheet" href="style.css" />
        </head>
        <main ref="main">
          {/* Hero styling */}
          <section className="section section-info section-shaped">
            {/* Background bubbles and spotify coloring */}
            <div className="shape shape-style-1 shape-default bg-gradient-spotify" style={{ borderBottom: "4px solid black" }}>
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
          </section>
          <div className="container-fluid"> {/* Container that houses are the components between the header and footer */}
            <div className="row" style={{ height: "600px" }}> {/* Sections within the container*/}
              {/* Radio buttons */}
              <div className="col-md-3">
                <div>
                  <h3 className="animate-character"> Select an Analysis</h3>
                  {/*<h2 class="wave" data-content="Select an Analysis">Select an Analysis</h2>*/}

                  <Form>
                    <FormGroup>
                      <FormGroup check>
                        <div className="custom-control custom-radio mb-3">
                          <input
                            className="custom-control-input"
                            id="customRadio5"
                            name="radio1"
                            type="radio"
                            value="graph"
                            onClick={this.handleRadioChange}
                          />
                          <label className="custom-control-label" htmlFor="customRadio5">
                            <strong>Emotional Analysis</strong>
                          </label>
                        </div>
                      </FormGroup>
                      <FormGroup check>
                        <div className="custom-control custom-radio mb-3">
                          <input
                            className="custom-control-input"
                            id="customRadio6"
                            name="radio1"
                            type="radio"
                            value="single"
                            onClick={this.handleRadioChange}
                          />
                          <label className="custom-control-label" htmlFor="customRadio6">
                            <strong>Single Song Analysis</strong>
                          </label>
                        </div>
                      </FormGroup>
                    </FormGroup>
                  </Form>
                </div>

                <h3 className="animate-character"> Specifications</h3> {/* Header */}
                {/* If graph radio button selected */}

                {selectedRadio === "graph" && (
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <Dropdown isOpen={dropdownOpen_Graph_TimeFrame} toggle={this.toggleDropdown_TimeFrame} style={{ marginTop: '10px' }}>
                      <DropdownToggle caret>{selectedTitle_TimeFrame || "Select Time Frame"}</DropdownToggle>
                      <DropdownMenu>
                        <DropdownItem onClick={() => this.handleDropdownSelect_TimeFrame('short_term')}>
                          4 Weeks
                        </DropdownItem>
                        <DropdownItem onClick={() => this.handleDropdownSelect_TimeFrame('medium_term')}>
                          6 Months
                        </DropdownItem>
                        <DropdownItem onClick={() => this.handleDropdownSelect_TimeFrame('long_term')}>
                          Overall
                        </DropdownItem>
                      </DropdownMenu>
                    </Dropdown>

                    {/* Map emotions array to dropdown */}
                    <Dropdown isOpen={dropdownOpen_Graph_SongRange} toggle={this.toggleDropdown_SongRange} style={{ marginTop: '50px' }}>
                      <DropdownToggle caret>{selectedTitle_SongRange || "Select Song Range"}</DropdownToggle>
                      <DropdownMenu style={{ maxHeight: '200px', overflowY: 'auto' }}>
                        {timeframe.map((timeframe) => (
                          <DropdownItem key={timeframe} onClick={() => this.handleDropdownSelect_SongRange(timeframe, 'dropdown2')}>
                            {timeframe}
                          </DropdownItem>
                        ))}
                      </DropdownMenu>
                    </Dropdown>

                    {/* Button to send data to server */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <Button
                        color="primary"
                        size="lg"
                        type="button"
                        className="ml-1"
                        style={{ marginTop: '50px' }}
                        onClick={this.sendGraphInput}
                      >
                        Send
                      </Button>
                    </div>
                  </div>
                )}
                {selectedRadio === "single" && (
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    
                    {/*Text Field for the single song*/}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <FormGroup>
                        <Input
                          id="singlesonginput"
                          placeholder="song name"
                          type="text"
                          onChange={this.handleSingleSongInputChange}
                        />
                      </FormGroup>
                    </div>

                    {/* Button to send data to server */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <Button
                        color="primary"
                        size="lg"
                        type="button"
                        className="ml-1"
                        style={{ marginTop: '25px' }}
                        onClick={this.sendSingleSongGraphInput}
                      >
                        Send
                      </Button>
                    </div>
                  </div> 
                )}
              </div>

             <div className="col-md-9" style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "600px", backgroundColor: "#F5F5F5" }}>
                {/* content of the col-md-9 */}
                {this.state.data ? (

                    <div className="charts" style={{display: "flex"}}>
                      <Card sx={{ minWidth: '25rem', minHeight: 250, maxHeight: 400, bgcolor: 'F5F5F5' }}>
                        <PieChart data={this.state.data}></PieChart>
                      </Card>

                      <Card sx={{ minWidth: '45rem', minHeight: 250, maxHeight: 400, bgcolor: 'F5F5F5' }}>
                        <TreeChart data={this.state.tree_data}></TreeChart>
                      </Card>
                    </div>
                ) : (

                  this.state.isButtonClicked && <p onClick={this.handleButtonClick}><div className="loader"></div></p>
                )}


              </div>
            </div>
          </div>
          <div className="TopSongs-Data">
            <div className="top-songs-label">

              <label className="custom-control-label" htmlFor="customRadio6">
                <h3 className="animate-character" style={{ marginLeft: '20px' }}>Your Top Songs</h3>
              </label>

            </div>

            {this.state.songs && (
              <SongsBox songs={this.state.songs}> </SongsBox>
            )}
          </div>
        </main>
        {/* <SimpleFooter /> Prebuilt Footer */}
      </>
    );
  }
}

export default Data;