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

const emotions = ["Admiration","Amusement","Anger","Annoyance","Approval","Caring","Confusion","Curiosity","Desire","Disappointment",'Disapproval','Disgust','Embarrassment',"Excitement","Fear","Gratitude","Grief","Joy","Love","Nervousness","Optimism","Pride","Realization","Relief","Remorse","Sadness","Surprise","Neutral"];
const timeframe =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

class Login extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedRadio: "",
      dropdownOpen: false
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

  render() {
    const { selectedRadio, dropdownOpen, dropdownOpen2 } = this.state;
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
                  <div>
                    
                    <Dropdown isOpen={dropdownOpen} toggle={this.toggleDropdown}>
                      <DropdownToggle caret>Select Time Frame</DropdownToggle>
                      <DropdownMenu>
                        <DropdownItem>4 Weeks</DropdownItem>
                        <DropdownItem>6 Months</DropdownItem>
                        <DropdownItem>Overall</DropdownItem>
                      </DropdownMenu>
                    </Dropdown>

                    <Dropdown isOpen={dropdownOpen2} toggle={this.toggleDropdown2} style={{ marginTop: '50px' }}>
                      <DropdownToggle caret>Select Song Range</DropdownToggle>
                      <DropdownMenu style={{ maxHeight: '200px', overflowY: 'auto'}}>
                      {timeframe.map((timeframe) => (
                              <DropdownItem key={timeframe}>{timeframe}</DropdownItem>
                        ))}
                    </DropdownMenu>
                    </Dropdown>
                  </div>
                  
                )}
  

{selectedRadio === "song" && (
  <div>
    <Dropdown isOpen={dropdownOpen} toggle={this.toggleDropdown}>
      <DropdownToggle caret>Select Emotion</DropdownToggle>
      <DropdownMenu style={{ maxHeight: '200px', overflowY: 'auto' }}>
        {emotions.map((emotion) => (
          <DropdownItem key={emotion}>{emotion}</DropdownItem>
        ))}
      </DropdownMenu>
    </Dropdown>
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