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
import { Button, Container, Row, Col } from "reactstrap";

// core components
import DemoNavbar from "components/Navbars/DemoNavbar.js";
import CardsFooter from "components/Footers/CardsFooter.js";

// index page sections
import Hero from "./IndexSections/Hero.js";
import Buttons from "./IndexSections/Buttons.js";
import Inputs from "./IndexSections/Inputs.js";
import CustomControls from "./IndexSections/CustomControls.js";
import Menus from "./IndexSections/Menus.js";
import Navbars from "./IndexSections/Navbars.js";
import Tabs from "./IndexSections/Tabs.js";
import Progress from "./IndexSections/Progress.js";
import Pagination from "./IndexSections/Pagination.js";
import Pills from "./IndexSections/Pills.js";
import Labels from "./IndexSections/Labels.js";
import Alerts from "./IndexSections/Alerts.js";
import Typography from "./IndexSections/Typography.js";
import Modals from "./IndexSections/Modals.js";
import Datepicker from "./IndexSections/Datepicker.js";
import TooltipPopover from "./IndexSections/TooltipPopover.js";
import Carousel from "./IndexSections/Carousel.js";
import Icons from "./IndexSections/Icons.js";
import Login from "./IndexSections/Login.js";
import Download from "./IndexSections/Download.js";
import SimpleFooter from "components/Footers/SimpleFooter.js";
import { authorize } from "util/SpotifyOath.js";

/* 
Class that represents the main components of
the webpage such as methods and properties
that define the appearence and behavior of
the webpage
*/
class Index extends React.Component {
  // This method ensures the webpage always starts at the top
  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }

  render() {
    return (
      // Nav bar for the top of the webpage
      <>
        <DemoNavbar />
        {/* HTML */}
        <main ref="main">
          {/* Positioning of elements */}
          <div className="position-relative">
            {/* Hero styling */}
            <section className="section section-info section-shaped">
              {/* Background squares */}
              <div className="shape shape-style-1 shape-default bg-gradient-spotify">
                <span className="span-150" />
                <span className="span-50" />
                <span className="span-50" />
                <span className="span-75" />
                <span className="span-100" />
                <span className="span-75" />
                <span className="span-50" />
                <span className="span-100" />
                <span className="span-50" />
                <span className="span-100" />
              </div>
              {/*
              Hero section: Imports various different components for the ceneter.
              This includes buttons, inputs, and alerts.
              */}
              {/* Creates a containers to hold a shape, centered vertically and horizontally */}
              <Container className="shape-container d-flex align-items-center py-lg">
                <div className="col px-0"> {/* column layout */}
                  <Row className="align-items-center justify-content-center"> {/* row layout */}
                    <Col className="text-center" lg="6"> {/* column layout */}
                      {/* image tag for logo */}
                      <img
                        alt="..."
                        className="img-fluid"
                        src={require("assets/img/brand/spotify-emotions-logo.png")}
                        style={{ width: "450px" }}
                      />
                      {/* Text */}
                      <p className="lead text-white">
                        A Natural Language Processing Application for
                        determining emotions based on recent
                        Spotify Listening Patterns.
                      </p>
                      {/* Button wrapper */}
                      <div className="btn-wrapper mt-5">
                        {/* Button with microphone logo */}
                        <Button
                          className="btn-yellow btn-icon mb-3 mb-sm-0"
                          color="default"
                          onClick={() =>
                            authorize()
                          }
                          size="lg"
                        >
                          <span className="btn-inner--icon mr-1">
                            <i className="fa fa-microphone" />
                          </span>
                          <span className="btn-inner--text">Try with Spotify</span>
                        </Button>{" "}
                      </div>
                    </Col>
                  </Row>
                </div>
              </Container>
              {/* Footer */}
              <div className="separator separator-bottom separator-skew zindex-100">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  preserveAspectRatio="none"
                  version="1.1"
                  viewBox="0 0 2560 100"
                  x="0"
                  y="0"
                >
                  <polygon
                    className="fill-white"
                    points="2560 0 2560 100 0 100"
                  />
                </svg>
              </div>
            </section>
          </div>
        </main>
        {/*Footer: <SimpleFooter />*/}
      </>
    );
  }
}

export default Index;
