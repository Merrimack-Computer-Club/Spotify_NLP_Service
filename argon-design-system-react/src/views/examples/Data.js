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

// reactstrap components
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

class Login extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedRadio: "",
      dropdownOpen: false,
      dropdownOpen2: false,
      dropdownOpen3: false,
      selectedVal: null,
      selectedVal2: null,
      selectedTitle: "",
      selectedTitle2: "",
      selectedVal3: null,
      selectedTitle3: ""
    };
  }



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

  
  sendSelectedInput = () => {
    const { selectedVal, selectedVal2 } = this.state;
    fetch('http://localhost:5000/api/save_input', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ data: [selectedVal, selectedVal2] })
    })
    .then(response => response.text())
    .then(data => console.log(data))
    .catch(error => console.error(error));
  }

  sendSelectedInput2= () => {
    const selectedInput3 = this.state.selectedVal3;
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
  
  
  
  handleDropdownSelect(value) {
    this.setState({ selectedVal: value, selectedTitle: value });
  }
  handleDropdownSelect2(value) {
    this.setState({ selectedVal2: value, selectedTitle2: value });
  }

  handleDropdownSelect3(value) {
    this.setState({ selectedVal3: value, selectedTitle3: value });
  }

  handleRadioChange = (event) => {
    this.setState({
      selectedRadio: event.target.value,
    });
  };

  toggleDropdown = () => {
    this.setState({
      dropdownOpen: !this.state.dropdownOpen,
    });
  };

  toggleDropdown2 = () => {
    this.setState({
      dropdownOpen2: !this.state.dropdownOpen2,
    });
  };
  toggleDropdown3 = () => {
    this.setState({
      dropdownOpen3: !this.state.dropdownOpen3,
    });
  };

  render() {
    const { selectedRadio, dropdownOpen, dropdownOpen2, dropdownOpen3, selectedVal, selectedTitle,selectedVal2, selectedTitle2, selectedTitle3, selectedVal3 } = this.state;
    return (
      <>
        <DemoNavbar />
        <head>
          <link rel="stylesheet" href="syle.css" />
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
  
          <div className="container-fluid">
            <div className="row" style={{ height: "600px" }}>
              {/* Radio buttons */}
              <div className="col-md-3">
                <div>
                  <h2>Select an Analysis</h2>
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
                          Graph
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
                          Song
                        </label>
                      </FormGroup>
                    </FormGroup>
                  </Form>
                </div>

                <h2>Choose:</h2>
                
                {selectedRadio === "graph" && (
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    
                    <Dropdown isOpen={dropdownOpen} toggle={this.toggleDropdown} style={{ marginTop: '10px' }}>
              <DropdownToggle caret>{selectedTitle || "Select Time Frame"}</DropdownToggle>
              <DropdownMenu>
                <DropdownItem onClick={() => this.handleDropdownSelect('4 Weeks')}>
                  4 Weeks
                </DropdownItem>
                <DropdownItem onClick={() => this.handleDropdownSelect('6 Months')}>
                  6 Months
                </DropdownItem>
                <DropdownItem onClick={() => this.handleDropdownSelect('Overall')}>
                  Overall
                </DropdownItem>
              </DropdownMenu>
            </Dropdown>

                    <Dropdown isOpen={dropdownOpen2} toggle={this.toggleDropdown2} style={{ marginTop: '50px' }}>
                      <DropdownToggle caret>{selectedTitle2 || "Select Song Range"}</DropdownToggle>
                      <DropdownMenu style={{ maxHeight: '200px', overflowY: 'auto' }}>
                      {timeframe.map((timeframe) => (
                                    <DropdownItem key={timeframe} onClick={() => this.handleDropdownSelect2(timeframe, 'dropdown2')}>
                                    {timeframe}
                                  </DropdownItem>
                        ))}
                    </DropdownMenu>
                    </Dropdown>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
<Button
                    color="primary"
                    size="lg"
                    type="button"
                    className="ml-1"
                    style={{ marginTop: '50px' }}
                    onClick={this.sendSelectedInput}
                  >
                    Send
                  </Button>
                
</div>
                  </div>
                  
                )}
  

{selectedRadio === "song" && (
  <div>
    <Dropdown isOpen={dropdownOpen3} toggle={this.toggleDropdown3}>
      <DropdownToggle caret>{selectedTitle3 || "Select an Emotion"}</DropdownToggle>
      <DropdownMenu style={{ maxHeight: '200px', overflowY: 'auto' }}>
        {emotions.map((emotion) => (
          <DropdownItem key={emotion} onClick={() => this.handleDropdownSelect3(emotion, 'dropdown3')}>
            {emotion}
          </DropdownItem>
        ))}
      </DropdownMenu>
    </Dropdown>
    <div style={{ display: 'flex', flexDirection: 'column' }}>
    <Button
                    color="primary"
                    size="lg"
                    type="button"
                    className="ml-1"
                    style={{ marginTop: '50px' }}
                    onClick={this.sendSelectedInput2}
                  >
                    Send
                  </Button>
                
</div>
  </div>
  
)}

              </div>
            </div>
          </div>
        </main>
        <SimpleFooter />
      </>
    );
  }
}

export default Login;