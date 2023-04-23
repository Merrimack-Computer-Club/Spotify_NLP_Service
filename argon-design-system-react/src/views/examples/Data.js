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
  Col
} from "reactstrap";

// core components
import DemoNavbar from "components/Navbars/DemoNavbar.js";
import SimpleFooter from "components/Footers/SimpleFooter.js";
import { getResponse, getProfile, getTopSongs } from "util/SpotifyOath.js";


class Login extends React.Component {
  constructor(props) {
    super(props);
    this.state = { color: "" };
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

  handleChange = event => {
    this.setState({ color: event.target.value });
  };

  render() {
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
                <h2>Select an Anaylsis</h2>
                <Form>
                  <FormGroup>
                    <FormGroup check>
                      <label check>
                        <Input
                          type="radio"
                          name="radio1"
                          value="green"
                          onClick={this.handleChange}
                        />
                        Graph
                      </label>
                    </FormGroup>
                    <FormGroup check>
                      <label check>
                        <Input
                          type="radio"
                          name="radio1"
                          value="blue"
                          onClick={this.handleChange}
                        />
                        Song
                      </label>
                    </FormGroup>
                  </FormGroup>
                </Form>
              </div>

             {/* Colored section */}
             <div
  className={
    this.state.color === "green" 
      ? "bg-green w-50" // add a width of 50%
      : this.state.color === "blue"
      ? "bg-blue w-25" // add a width of 75%
      : "bg-green w-75" // Default color with a width of 100%
  }
>

                <div className="container">
                  <div className="row">
                    <div className="col-md-5">
                      <h2>Results</h2>
                    </div>
                  </div>
                </div>
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