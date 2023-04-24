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
  Card,
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

// core components
import DemoNavbar from "components/Navbars/DemoNavbar.js";
import SimpleFooter from "components/Footers/SimpleFooter.js";
import { getResponse, getProfile, getTopSongs } from "util/SpotifyOath.js";

// List of emotion to select from dropdown
const emotions = ["Admiration","Amusement","Anger","Annoyance","Approval","Caring","Confusion","Curiosity","Desire","Disappointment",'Disapproval','Disgust','Embarrassment',"Excitement","Fear","Gratitude","Grief","Joy","Love","Nervousness","Optimism","Pride","Realization","Relief","Remorse","Sadness","Surprise","Neutral"];
// List of time ranges to select from dropdown
const timeframe =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

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
      selectedTitle_Emotion: ""
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
      getTopSongs();
    });
  }
  
  /*
  This method is responsible for creating 
  post requests of the graph data. Graph data
  contains info on Time Frame & Song Range the
  user has selected from the dropdown menus.
  */
  sendGraphInput = () => {
    const { selectedVal_TimeFrame, selectedVal_SongRange } = this.state;
    fetch('http://localhost:5000/api/save_input', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ data: [selectedVal_TimeFrame, selectedVal_SongRange] })
    })
    .then(response => response.text())
    .then(data => console.log(data))
    .catch(error => console.error(error));
  }

  /*
  This method is responsible for creating 
  post requests of the song rec. data. Song rec. data
  contains info on Emotion the
  user has selected from the dropdown menu.
  */
  sendSongRecommendationInput= () => {
    const selectedInput3 = this.state.selectedVal_Emotion;
    fetch('http://localhost:5000/api/save_input', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ data: selectedInput3})
    })
    .then(response => response.text())
    .then(data => console.log(data))
    .catch(error => console.error(error));
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
        <section className="section section-shaped section-lg">
          <div className="shape shape-style-3 bg-gradient-default">
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
          <div className="row" style={{ height: "600px"}}> {/* Sections within the container*/}
            {/* Radio buttons */}
            <div className="col-md-3"  style={{ height: "600px"}}> 
              <div>
              <h2 class="title">
                <span class="title-word title-word-1">Select</span>
                <span class="title-word title-word-2">an</span>
                <span class="title-word title-word-3">Analysis</span>
              </h2>
              {/*<h2 class="wave" data-content="Select an Analysis">Select an Analysis</h2>*/}

                <Form>
                  <FormGroup>
                    <FormGroup check>
                      <label check>
                        <Input
                          type="radio"
                          name="radio1"
                          value="graph"
                          onClick={this.handleRadioChange} 
                        />
                        Emotional Anaylsis
                      </label>
                    </FormGroup>
                    <FormGroup check>
                      <label check>
                        <Input
                          type="radio"
                          name="radio1"
                          value="song"
                          onClick={this.handleRadioChange}
                        />
                        Song Recommendation
                      </label>
                    </FormGroup>
                  </FormGroup>
                </Form>
              </div>
 
              <h2 class="title">
                <span class="title-word title-word-1">Specifitcations</span>
              </h2> {/* Header */}
             {/* If graph radio button selected */}

            {selectedRadio === "graph" && (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <Dropdown isOpen={dropdownOpen_Graph_TimeFrame} toggle={this.toggleDropdown_TimeFrame} style={{ marginTop: '10px' }}>
                  <DropdownToggle caret>{selectedTitle_TimeFrame || "Select Time Frame"}</DropdownToggle>
                  <DropdownMenu>
                    <DropdownItem onClick={() => this.handleDropdownSelect_TimeFrame('4 Weeks')}>
                      4 Weeks
                    </DropdownItem>
                    <DropdownItem onClick={() => this.handleDropdownSelect_TimeFrame('6 Months')}>
                      6 Months
                    </DropdownItem>
                    <DropdownItem onClick={() => this.handleDropdownSelect_TimeFrame('Overall')}>
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
  
             {/* If song radio button selected */}       
             {selectedRadio === "song" && (
              <div>
                {/* Map song array to dropdown */}
                <Dropdown isOpen={dropdownOpen_Song_Emotion} toggle={this.toggleDropdown_Emotion}>
                  <DropdownToggle caret>{selectedTitle_Emotion || "Select an Emotion"}</DropdownToggle>
                  <DropdownMenu style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {emotions.map((emotion) => (
                      <DropdownItem key={emotion} onClick={() => this.handleDropdownSelect_Emotion(emotion, 'dropdown3')}>
                        {emotion}
                      </DropdownItem>
                    ))}
                  </DropdownMenu>
                </Dropdown>

                {/* Button to send emotion data */}
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <Button
                    color="primary"
                    size="lg"
                    type="button"
                    className="ml-1"
                    style={{ marginTop: '50px' }}
                    onClick={this.sendSongRecommendationInput}
                  >
                    Send
                  </Button>
                </div>
              </div>
             )}

             </div>

             <div className="col-md-9" style={{ height: "600px" , backgroundColor: "rgb(196, 194, 187)"}}>
      {/* content of the col-md-9 */}
    </div>
          
          </div>
        </div>
      </main>
        <SimpleFooter /> {/* Prebuilt Footer */}
      </>
      );
    }
  }

export default Data;